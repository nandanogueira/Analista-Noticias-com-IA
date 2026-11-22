"""Agente que orquestra: pergunta -> LLM -> ferramenta (NewsAPI) -> LLM -> resposta.

Fluxo:
1. Valida a pergunta do usuário.
2. Envia a pergunta + definição das ferramentas ao LLM.
3. O LLM interpreta a intenção e decide qual ferramenta chamar (ou nenhuma).
4. A aplicação executa a ferramenta (consulta real à NewsAPI) e processa o retorno.
5. O resultado volta ao LLM, que pode refinar a busca ou redigir a resposta final.
6. A resposta é devolvida com as notícias (fontes) e o registro das ferramentas usadas.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .config import HISTORY_TURNS, MAX_QUESTION_LENGTH, MAX_TOOL_ROUNDS, MIN_QUESTION_LENGTH
from .errors import LLMError, ValidationError
from .llm_client import LLMClient
from .news_api import NewsAPIClient
from .processing import Article
from .tools import TOOL_DEFINITIONS, TOOL_LABELS, ToolCallRecord, ToolExecutor

logger = logging.getLogger(__name__)

_WEEKDAYS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]

SYSTEM_PROMPT = """Você é o "Analista de Notícias", um assistente que responde perguntas sobre atualidades \
usando EXCLUSIVAMENTE notícias reais obtidas pelas ferramentas disponíveis. Hoje é {hoje}.

## Como agir
1. Interprete a pergunta: identifique o assunto, palavras-chave, período, país e idioma desejados.
2. Escolha a ferramenta adequada:
   - buscar_noticias: perguntas sobre um assunto específico (tema, pessoa, empresa, evento, lugar).
     Monte uma consulta com palavras-chave curtas (não a frase inteira). Use idioma "pt" por padrão;
     se o assunto for internacional ou o usuário pedir, use "en" ou "todos".
     Converta períodos relativos ("esta semana", "ontem") em datas AAAA-MM-DD.
   - manchetes_principais: perguntas gerais ("o que está acontecendo hoje?", "destaques de tecnologia").
   - listar_fontes: quando o usuário perguntar quais veículos/fontes estão disponíveis.
3. Se a busca não retornar notícias, tente UMA nova busca com termos mais amplos, sinônimos ou outro idioma.
4. Se a pergunta NÃO for sobre notícias ou atualidades (ex.: cumprimento, matemática, receita),
   não use ferramentas: responda em 1-2 frases explicando que você analisa notícias e sugira uma pergunta.

## Regras para a resposta
- Responda em português do Brasil, de forma clara e objetiva.
- Use SOMENTE as informações dos artigos retornados. Nunca invente fatos, números, datas ou fontes.
- Cite as notícias pelo id entre colchetes, ex.: [1] ou [2][5], logo após cada afirmação.
- Se as fontes divergirem, aponte a divergência. Se as notícias forem insuficientes, diga isso.
- NÃO liste as fontes/links no final: o sistema exibe as fontes automaticamente.
- Use exatamente esta estrutura em Markdown:

### 📝 Resumo
(2 a 4 frases respondendo diretamente à pergunta)

### 📌 Principais acontecimentos
- (um tópico por acontecimento, com data quando relevante e citações [n])

### 🔍 Contexto e análise
(explique por que isso importa, causas, consequências ou tendências, com base nas notícias)

### ⚠️ Observações
(limitações: período coberto, poucas fontes, divergências; a NewsAPI gratuita tem atraso de ~24h)
"""

_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class AgentResponse:
    question: str
    answer: str
    articles: list[Article] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    rounds: int = 0
    elapsed_s: float = 0.0
    # Motivos pelos quais modelos anteriores falharam antes deste responder
    fallback_notes: list[str] = field(default_factory=list)


def validate_question(question: str | None) -> str:
    if question is None or not str(question).strip():
        raise ValidationError("Digite uma pergunta sobre algum assunto do noticiário.")
    question = " ".join(str(question).split())
    if len(question) < MIN_QUESTION_LENGTH:
        raise ValidationError("A pergunta está muito curta. Descreva o assunto que deseja analisar.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValidationError(
            f"A pergunta está muito longa ({len(question)} caracteres). "
            f"O limite é {MAX_QUESTION_LENGTH} caracteres."
        )
    return question


def build_system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    hoje = f"{_WEEKDAYS[today.weekday()]}, {today:%d/%m/%Y} ({today.isoformat()})"
    return SYSTEM_PROMPT.format(hoje=hoje)


class NewsAgent:
    def __init__(
        self,
        llms: LLMClient | list[LLMClient],
        news: NewsAPIClient,
        max_rounds: int = MAX_TOOL_ROUNDS,
    ):
        """``llms`` em ordem de prioridade: o primeiro é o principal e os demais são reservas."""
        self.llms = list(llms) if isinstance(llms, (list, tuple)) else [llms]
        if not self.llms:
            raise ValueError("Informe ao menos um modelo de linguagem.")
        self.news = news
        self.max_rounds = max_rounds

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Responde à pergunta; se um modelo falhar, refaz tudo com o próximo da lista."""
        notify = on_event or (lambda _msg: None)
        started = time.perf_counter()
        question = validate_question(question)

        failures: list[str] = []
        for index, llm in enumerate(self.llms):
            try:
                response = self._run(llm, question, history, notify, started)
            except LLMError as exc:
                failures.append(f"{llm.provider.label} ({llm.model}): {exc.user_message}")
                logger.warning("Modelo %s falhou: %s | %s", llm.provider.label, exc.user_message, exc.detail)
                if index + 1 < len(self.llms):
                    notify(
                        f"⚠️ {llm.provider.label} falhou ({exc.user_message}) "
                        f"Trocando para {self.llms[index + 1].provider.label}..."
                    )
                    continue
                if len(self.llms) == 1:
                    raise
                raise LLMError(
                    "Nenhum modelo de IA conseguiu responder. " + " | ".join(failures),
                    detail=exc.detail,
                    code=exc.code,
                ) from exc
            response.fallback_notes = failures
            return response
        raise AssertionError("inalcançável")  # pragma: no cover

    def _run(
        self,
        llm: LLMClient,
        question: str,
        history: list[dict[str, str]] | None,
        notify: Callable[[str], None],
        started: float,
    ) -> AgentResponse:
        """Executa o ciclo IA -> ferramenta -> IA com um modelo específico."""
        executor = ToolExecutor(self.news)

        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
        messages.extend(_history_messages(history))
        messages.append({"role": "user", "content": question})

        for round_number in range(1, self.max_rounds + 1):
            final_round = round_number == self.max_rounds
            notify("🧠 A IA está interpretando a pergunta..." if round_number == 1 else "🧠 A IA está analisando os resultados...")

            message = llm.chat(
                messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="none" if final_round else "auto",
            )
            tool_calls = message.get("tool_calls") or []

            if tool_calls and not final_round:
                tool_calls = _ensure_tool_call_ids(tool_calls, round_number)
                messages.append(_assistant_message(message, tool_calls))
                for call in tool_calls:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    notify(f"🔧 Ferramenta escolhida pela IA: **{TOOL_LABELS.get(name, name)}**")
                    result = executor.execute(name, function.get("arguments"))
                    record = executor.records[-1]
                    notify(f"📰 {record.summary or record.error}")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                continue

            answer = (message.get("content") or "").strip()
            answer = _strip_reasoning(answer)
            if not answer:
                raise LLMError(
                    "O modelo de IA não gerou uma resposta. Tente reformular a pergunta ou trocar de modelo."
                )
            notify("✍️ Resposta gerada.")
            return AgentResponse(
                question=question,
                answer=answer,
                articles=executor.articles,
                tool_calls=executor.records,
                provider=llm.provider.label,
                model=llm.model,
                rounds=round_number,
                elapsed_s=round(time.perf_counter() - started, 1),
            )

        raise LLMError("A IA excedeu o número máximo de etapas sem chegar a uma resposta.")


# ---------------------------------------------------------------------- #
def _history_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Mantém as últimas trocas (sem citações antigas) para perguntas de acompanhamento."""
    result = []
    for turn in (history or [])[-HISTORY_TURNS * 2 :]:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            result.append({"role": role, "content": _CITATION_RE.sub("", content)[:2000]})
    return result


def _ensure_tool_call_ids(tool_calls: list[dict[str, Any]], round_number: int) -> list[dict[str, Any]]:
    fixed = []
    for i, call in enumerate(tool_calls):
        call = dict(call)
        call.setdefault("type", "function")
        if not call.get("id"):
            call["id"] = f"call_{round_number}_{i}"
        fixed.append(call)
    return fixed


def _assistant_message(message: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Reenvia a mensagem do assistente só com campos padrão.

    Campos extras da resposta (ex.: "reasoning" do Groq) são descartados, mas o
    Gemini exige que "extra_content" (assinatura de pensamento) seja mantido
    dentro de cada tool_call, por isso os tool_calls são preservados inteiros.
    """
    return {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}


def _strip_reasoning(text: str) -> str:
    """Remove blocos <think>...</think> que alguns modelos (ex.: Qwen) incluem."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
