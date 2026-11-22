"""Testes unitários das partes determinísticas (validação, processamento, citações).

Rodar com:  python -m pytest -q
(As consultas reais à NewsAPI e ao LLM são feitas pela aplicação, não aqui.)
"""
import json
from datetime import date, timedelta

import pytest

from src.agent import _assistant_message, _strip_reasoning, build_system_prompt, validate_question
from src.errors import NewsAPIError, ToolArgumentError, ValidationError
from src.formatting import extract_citations, link_citations
from src.processing import Article, clean_text, process_articles
from src.tools import ToolExecutor, _page_size, _parse_arguments


def test_validate_question():
    assert validate_question("  notícias   sobre  IA ") == "notícias sobre IA"
    with pytest.raises(ValidationError):
        validate_question("   ")
    with pytest.raises(ValidationError):
        validate_question("x" * 501)


def test_process_articles_filters_and_dedupes():
    raw = [
        {"title": "[Removed]", "url": "https://removed.com"},
        {"title": "IA avança", "url": "https://a.com/1", "source": {"name": "A"},
         "publishedAt": "2026-11-20T10:30:00Z", "content": "Texto <b>importante</b> [+1234 chars]"},
        {"title": "IA avança!", "url": "https://b.com/1", "source": {"name": "B"}},  # mesmo título
        {"title": "Outra", "url": "https://a.com/1"},  # mesma URL
        {"title": "Nova", "url": "https://c.com/2", "source": {"name": "C"}},
    ]
    arts = process_articles(raw, start_id=5)
    assert [a.id for a in arts] == [5, 6]
    assert arts[0].published_at == "20/11/2026 10:30 UTC"
    assert arts[0].content == "Texto importante"


def test_clean_text_truncates():
    assert clean_text("palavra " * 100, 30).endswith("…")


def test_citations():
    arts = [Article(id=1, title="t", source="G1", url="https://g1.com/x", published_at="")]
    answer = "Fato importante [1]. Outro [1, 2]."
    assert extract_citations(answer) == [1, 2]
    linked = link_citations(answer, arts)
    assert "[[1]](https://g1.com/x" in linked
    assert "[2]" in linked  # id inexistente permanece como texto


def test_parse_arguments():
    assert _parse_arguments('{"consulta": "IA"}') == {"consulta": "IA"}
    assert _parse_arguments(None) == {}
    with pytest.raises(ToolArgumentError):
        _parse_arguments("{quebrado")


def test_page_size_clamped():
    assert _page_size(None) == 10
    assert _page_size(999) == 20
    assert _page_size("abc") == 10


class FakeNewsClient:
    """Dublê de teste do cliente HTTP (usado só nos testes unitários)."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def everything(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"status": "ok", "totalResults": 1,
                "articles": [{"title": "T", "url": "https://x.com", "source": {"name": "X"}}]}


def test_executor_adjusts_old_date_and_returns_payload():
    news = FakeNewsClient()
    ex = ToolExecutor(news)
    old = (date.today() - timedelta(days=90)).isoformat()
    payload = json.loads(ex.execute("buscar_noticias", {"consulta": "IA", "data_inicial": old}))
    assert payload["retornados"] == 1
    assert payload["avisos"]
    assert news.calls[0]["from_date"] != old


def test_executor_returns_recoverable_errors_to_llm():
    ex = ToolExecutor(FakeNewsClient())
    payload = json.loads(ex.execute("ferramenta_inexistente", {}))
    assert "erro" in payload
    payload = json.loads(ex.execute("buscar_noticias", {"consulta": "IA", "idioma": "xx"}))
    assert "erro" in payload


def test_executor_raises_fatal_errors():
    ex = ToolExecutor(FakeNewsClient(error=NewsAPIError("chave inválida", code="apiKeyInvalid")))
    with pytest.raises(NewsAPIError):
        ex.execute("buscar_noticias", {"consulta": "IA"})


def test_assistant_message_keeps_only_standard_fields():
    msg = {"role": "assistant", "content": None, "reasoning": "...", "tool_calls": []}
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"},
              "extra_content": {"google": {"thought_signature": "abc"}}}]
    out = _assistant_message(msg, calls)
    assert "reasoning" not in out
    assert out["tool_calls"][0]["extra_content"]


class FakeLLM:
    """Dublê de teste do cliente de IA (usado só nos testes unitários)."""

    def __init__(self, provider, error=None, answer="### 📝 Resumo\nOk [1]"):
        self.provider = provider
        self.model = "modelo-teste"
        self.error = error
        self.answer = answer
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls += 1
        if self.error:
            raise self.error
        return {"role": "assistant", "content": self.answer}


def test_fallback_to_backup_model_on_rate_limit():
    from src.agent import NewsAgent
    from src.config import GEMINI, GROQ
    from src.errors import LLMError

    primary = FakeLLM(GROQ, error=LLMError("Limite de uso gratuito atingido.", code="http429"))
    backup = FakeLLM(GEMINI)
    notes = []
    response = NewsAgent([primary, backup], FakeNewsClient()).ask("Notícias sobre IA", on_event=notes.append)

    assert (primary.calls, backup.calls) == (1, 1)
    assert response.provider == GEMINI.label
    assert "Groq" in response.fallback_notes[0] and "Limite" in response.fallback_notes[0]
    assert any("Trocando para" in n for n in notes)


def test_primary_model_used_when_healthy():
    from src.agent import NewsAgent
    from src.config import GEMINI, GROQ

    primary, backup = FakeLLM(GROQ), FakeLLM(GEMINI)
    response = NewsAgent([primary, backup], FakeNewsClient()).ask("Notícias sobre IA")
    assert backup.calls == 0
    assert response.provider == GROQ.label and response.fallback_notes == []


def test_all_models_failing_raises_combined_error():
    from src.agent import NewsAgent
    from src.config import GEMINI, GROQ
    from src.errors import LLMError

    agent = NewsAgent(
        [FakeLLM(GROQ, error=LLMError("cota Groq", code="http429")),
         FakeLLM(GEMINI, error=LLMError("cota Gemini", code="http429"))],
        FakeNewsClient(),
    )
    with pytest.raises(LLMError) as exc:
        agent.ask("Notícias sobre IA")
    assert "cota Groq" in exc.value.user_message and "cota Gemini" in exc.value.user_message


def test_invalid_question_does_not_call_any_model():
    from src.agent import NewsAgent
    from src.config import GROQ

    llm = FakeLLM(GROQ)
    with pytest.raises(ValidationError):
        NewsAgent([llm], FakeNewsClient()).ask("  ")
    assert llm.calls == 0


def test_misc():
    assert _strip_reasoning("<think>x</think>Olá") == "Olá"
    assert "2026-11-22" in build_system_prompt(date(2026, 11, 22))
