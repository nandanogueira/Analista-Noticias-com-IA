"""Interface Streamlit do Analista de Notícias com IA.

Execute com:  streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import traceback

import streamlit as st

from src.agent import AgentResponse, NewsAgent
from src.config import MAX_QUESTION_LENGTH, PROVIDERS, get_setting
from src.errors import AppError, ConfigError
from src.formatting import escape_markdown_link_text, extract_citations, link_citations
from src.llm_client import LLMClient
from src.news_api import NewsAPIClient
from src.processing import Article
from src.tools import TOOL_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Analista de Notícias com IA", page_icon="📰", layout="centered")

EXAMPLES = [
    "Quais as últimas notícias sobre inteligência artificial no Brasil?",
    "O que aconteceu com a economia brasileira nesta semana?",
    "Quais são as principais manchetes de tecnologia nos EUA hoje?",
    "Quais fontes de notícias em português estão disponíveis?",
]

# ---------------------------------------------------------------------- #
# Estado da sessão
# ---------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])  # [{role, content, response?, error?, detail?}]
st.session_state.setdefault("pending_question", None)


# ---------------------------------------------------------------------- #
# Configuração (somente via .env / variáveis de ambiente / st.secrets)
# ---------------------------------------------------------------------- #
def load_settings() -> dict:
    llms = [
        {
            "provider": provider,
            "model": get_setting(provider.model_env) or provider.default_model,
            "key": get_setting(provider.key_env),
        }
        for provider in PROVIDERS
    ]
    return {"llms": llms, "news_key": get_setting("NEWSAPI_KEY"), "show_details": True}


def build_llm_clients(llms: list[dict]) -> list[LLMClient]:
    """Cria os clientes de IA (principal primeiro) para os provedores que têm chave."""
    configured = [cfg for cfg in llms if cfg["key"]]
    if not configured:
        names = " ou ".join(cfg["provider"].key_env for cfg in llms)
        raise ConfigError(
            f"Nenhuma chave de IA configurada. Defina {names} no arquivo .env e reinicie a aplicação."
        )
    # Só o último da fila repete tentativas em caso de erro; os demais falham
    # rápido para que a reserva assuma sem espera.
    return [
        LLMClient(cfg["provider"], cfg["key"], cfg["model"], max_retries=2 if i == len(configured) - 1 else 0)
        for i, cfg in enumerate(configured)
    ]


# ---------------------------------------------------------------------- #
# Renderização
# ---------------------------------------------------------------------- #
def render_article(article: Article, cited: bool) -> None:
    with st.container(border=True):
        title = escape_markdown_link_text(article.title)
        badge = "⭐ " if cited else ""
        st.markdown(f"{badge}**[{article.id}] [{title}]({article.url})**")
        meta = " · ".join(x for x in (article.source, article.published_at, article.author) if x)
        st.caption(meta)
        if article.description:
            st.write(article.description)


def render_tool_details(response: AgentResponse) -> None:
    with st.expander("🔍 Como a IA chegou a esta resposta", expanded=False):
        st.markdown(
            f"**Modelo:** `{response.model}` ({response.provider}) · "
            f"**Etapas:** {response.rounds} · **Tempo total:** {response.elapsed_s}s"
        )
        if not response.tool_calls:
            st.info("A IA entendeu que a pergunta não exigia consulta à API de notícias.")
            return
        for i, call in enumerate(response.tool_calls, start=1):
            status = "✅" if call.ok else "❌"
            st.markdown(f"**{i}. {status} {TOOL_LABELS.get(call.name, call.name)}** (`{call.name}`) · {call.elapsed_ms} ms")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Argumentos escolhidos pela IA")
                st.code(json.dumps(call.arguments, ensure_ascii=False, indent=2), language="json")
            with col2:
                st.caption("Requisição enviada à NewsAPI")
                params = {k: v for k, v in call.api_params.items() if v not in (None, "")}
                st.code(json.dumps(params, ensure_ascii=False, indent=2), language="json")
            if call.summary:
                st.caption(f"Resultado: {call.summary}")
            for warning in call.warnings:
                st.warning(warning)
            if call.error:
                st.error(f"Erro na ferramenta: {call.error}")


def render_response(response: AgentResponse, show_details: bool) -> None:
    if response.fallback_notes:
        st.warning(
            f"⚠️ O modelo principal não respondeu; esta resposta foi gerada por **{response.provider}** "
            f"(`{response.model}`).\n\n" + "\n".join(f"- {note}" for note in response.fallback_notes)
        )
    st.markdown(link_citations(response.answer, response.articles))

    cited_ids = extract_citations(response.answer)
    cited = [a for a in response.articles if a.id in cited_ids]
    others = [a for a in response.articles if a.id not in cited_ids]

    if response.articles:
        st.markdown(f"#### 📚 Fontes citadas ({len(cited)})" if cited else "#### 📚 Notícias encontradas")
        for article in cited or others:
            render_article(article, cited=bool(cited))
        if cited and others:
            with st.expander(f"Outras {len(others)} notícias encontradas (não citadas)"):
                for article in others:
                    render_article(article, cited=False)

    sources = [s for call in response.tool_calls for s in call.sources]
    if sources:
        st.markdown(f"#### 🏢 Fontes disponíveis ({len(sources)})")
        st.dataframe(
            [{k: s[k] for k in ("nome", "categoria", "idioma", "pais", "url")} for s in sources],
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("url")},
        )

    if show_details:
        render_tool_details(response)


def render_error(message: dict, show_details: bool) -> None:
    st.error(message["error"])
    if show_details and message.get("detail"):
        with st.expander("Detalhes técnicos do erro"):
            st.code(message["detail"], language=None)


def render_history(show_details: bool) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📰"):
            if message.get("response"):
                render_response(message["response"], show_details)
            elif message.get("error"):
                render_error(message, show_details)
            else:
                st.markdown(message["content"])


# ---------------------------------------------------------------------- #
# Processamento de uma pergunta
# ---------------------------------------------------------------------- #
def handle_question(question: str, settings: dict) -> None:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("content") and not m.get("error")
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📰"):
        try:
            llms = build_llm_clients(settings["llms"])
            news = NewsAPIClient(settings["news_key"])
            agent = NewsAgent(llms, news)

            with st.status("Processando sua pergunta...", expanded=True) as status:
                response = agent.ask(question, history=history, on_event=st.write)
                status.update(label="Análise concluída", state="complete", expanded=False)

            render_response(response, settings["show_details"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response.answer, "response": response}
            )
        except AppError as exc:
            kind = "config" if isinstance(exc, ConfigError) else "app"
            logging.warning("Erro tratado (%s): %s | %s", kind, exc.user_message, exc.detail)
            message = {"role": "assistant", "content": "", "error": exc.user_message, "detail": exc.detail}
            render_error(message, settings["show_details"])
            st.session_state.messages.append(message)
        except Exception as exc:  # rede de segurança: nunca quebrar a interface
            logging.exception("Erro inesperado")
            message = {
                "role": "assistant",
                "content": "",
                "error": "Ocorreu um erro inesperado ao processar sua pergunta. Tente novamente.",
                "detail": f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            }
            render_error(message, settings["show_details"])
            st.session_state.messages.append(message)


# ---------------------------------------------------------------------- #
# Página
# ---------------------------------------------------------------------- #
def main() -> None:
    settings = load_settings()

    title_col, clear_col = st.columns([5, 2], vertical_alignment="center")
    title_col.title("📰 Analista de Notícias com IA")
    if clear_col.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Pergunte sobre qualquer assunto. A IA interpreta sua pergunta, escolhe a ferramenta de busca, "
        "consulta a NewsAPI em tempo real e resume os principais acontecimentos com as fontes."
    )
    active = [f"{c['provider'].label} (`{c['model']}`)" for c in settings["llms"] if c["key"]]
    if active:
        st.caption("🤖 IA: " + " → ".join(active) + (" · a reserva assume se a principal falhar" if len(active) > 1 else ""))

    if not st.session_state.messages and not st.session_state.pending_question:
        st.markdown("**Experimente uma pergunta:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    render_history(settings["show_details"])

    typed = st.chat_input(
        "Ex.: O que está acontecendo com as eleições nos EUA?", max_chars=MAX_QUESTION_LENGTH
    )
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if question:
        handle_question(question, settings)

    st.caption(
        "Notícias: [NewsAPI](https://newsapi.org) · plano gratuito com atraso de ~24h e busca de até 1 mês. "
        "Resumos gerados por IA podem conter imprecisões: confira as fontes."
    )


main()
