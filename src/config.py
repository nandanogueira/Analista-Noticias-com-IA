"""Configurações centrais da aplicação.

As chaves são lidas, nesta ordem, de:
1. variáveis de ambiente / arquivo ``.env``;
2. ``st.secrets`` (quando publicado no Streamlit Community Cloud).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    key: str
    label: str
    base_url: str
    key_env: str  # nome da variável de ambiente com a chave de API
    model_env: str  # nome da variável de ambiente com o modelo (opcional)
    default_model: str
    signup_url: str


# Groq e Gemini expõem endpoints compatíveis com o formato "chat/completions"
# da OpenAI, incluindo function calling (tools), então um único cliente HTTP
# simples feito com ``requests`` atende aos dois.
GROQ = ProviderConfig(
    key="groq",
    label="Groq",
    base_url="https://api.groq.com/openai/v1",
    key_env="GROQ_API_KEY",
    model_env="GROQ_MODEL",
    default_model="openai/gpt-oss-120b",
    signup_url="https://console.groq.com/keys",
)
GEMINI = ProviderConfig(
    key="gemini",
    label="Google Gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    key_env="GEMINI_API_KEY",
    model_env="GEMINI_MODEL",
    default_model="gemini-3-flash-preview",
    signup_url="https://aistudio.google.com/apikey",
)

# Ordem de prioridade: o primeiro é o modelo principal; se falhar (limite de
# cota, instabilidade, timeout...), a pergunta é refeita com o seguinte.
PROVIDERS: list[ProviderConfig] = [GROQ, GEMINI]

# Limites e parâmetros gerais
MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500
MAX_TOOL_ROUNDS = 4          # nº máximo de rodadas IA -> ferramenta -> IA
DEFAULT_PAGE_SIZE = 10       # notícias por consulta
MAX_PAGE_SIZE = 20
HISTORY_TURNS = 3            # pares pergunta/resposta enviados como contexto
NEWSAPI_MAX_DAYS_BACK = 29   # plano gratuito só busca ~1 mês para trás
LLM_TIMEOUT = 90
NEWSAPI_TIMEOUT = 15


def get_setting(name: str | None, default: str | None = None) -> str | None:
    """Lê uma configuração do ambiente ou de ``st.secrets``."""
    if not name:
        return default
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:  # st.secrets lança exceção se não houver secrets.toml
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default
