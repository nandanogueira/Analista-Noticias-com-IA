"""Exceções da aplicação.

Toda exceção carrega uma mensagem amigável (``user_message``) que pode ser
exibida diretamente na interface, e um ``detail`` técnico opcional, exibido
apenas no painel de detalhes.
"""
from __future__ import annotations


class AppError(Exception):
    """Erro base com mensagem amigável para o usuário."""

    def __init__(self, user_message: str, detail: str | None = None, code: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
        self.code = code


class ConfigError(AppError):
    """Configuração ausente ou inválida (ex.: chave de API não definida)."""


class ValidationError(AppError):
    """Entrada do usuário inválida (ex.: pergunta vazia ou longa demais)."""


class NewsAPIError(AppError):
    """Falha ao consultar a NewsAPI."""

    # Erros que não adianta a IA tentar contornar: interrompem o fluxo
    # e são mostrados direto ao usuário.
    FATAL_CODES = {
        "apiKeyMissing",
        "apiKeyInvalid",
        "apiKeyDisabled",
        "apiKeyExhausted",
        "rateLimited",
        "connection",
        "timeout",
    }

    @property
    def fatal(self) -> bool:
        return self.code in self.FATAL_CODES


class LLMError(AppError):
    """Falha ao consultar o modelo de linguagem."""


class ToolArgumentError(AppError):
    """A IA chamou uma ferramenta com argumentos inválidos."""
