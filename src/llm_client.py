"""Cliente HTTP para modelos de linguagem com API compatível com OpenAI.

Usa apenas ``requests`` (POST em ``/chat/completions``), então funciona com
Groq e Google Gemini (endpoints compatíveis com OpenAI, com function calling).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .config import LLM_TIMEOUT, ProviderConfig
from .errors import ConfigError, LLMError

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    def __init__(
        self,
        provider: ProviderConfig,
        api_key: str | None,
        model: str,
        timeout: int = LLM_TIMEOUT,
        max_retries: int = 2,
    ):
        if provider.key_env and not api_key:
            raise ConfigError(
                f"Chave de API do provedor {provider.label} não configurada. "
                f"Defina {provider.key_env} no arquivo .env e reinicie a aplicação. "
                f"Crie uma em {provider.signup_url}"
            )
        if not model or not model.strip():
            raise ConfigError("Nome do modelo de linguagem vazio. Defina GROQ_MODEL / GEMINI_MODEL no .env ou remova a variável para usar o padrão.")
        self.provider = provider
        self.api_key = api_key
        self.model = model.strip()
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Envia a conversa ao modelo e devolve a mensagem de resposta."""
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        data = self._request("POST", "/chat/completions", body)
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "O modelo de IA retornou uma resposta em formato inesperado.",
                detail=str(data)[:500],
            ) from exc

    def list_models(self) -> list[str]:
        data = self._request("GET", "/models")
        items = data.get("data") or data.get("models") or []
        names = []
        for item in items:
            name = item.get("id") or item.get("name") if isinstance(item, dict) else None
            if name:
                names.append(name.removeprefix("models/"))
        return sorted(set(names))

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.provider.base_url.rstrip("/") + path
        last_error: LLMError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(
                    method, url, json=body, headers=self._headers(), timeout=self.timeout
                )
            except requests.Timeout as exc:
                last_error = LLMError(
                    "O modelo de IA demorou demais para responder. Tente novamente.",
                    detail=str(exc),
                    code="timeout",
                )
                continue
            except requests.ConnectionError as exc:
                raise LLMError(
                    f"Não foi possível conectar ao {self.provider.label}. Verifique sua conexão com a internet.",
                    detail=str(exc),
                    code="connection",
                ) from exc
            except requests.RequestException as exc:
                raise LLMError("Falha inesperada ao chamar o modelo de IA.", detail=str(exc)) from exc

            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError(
                        "O provedor de IA retornou uma resposta inválida.", detail=response.text[:300]
                    ) from exc

            last_error = self._http_error(response)
            retryable = response.status_code in _RETRY_STATUS or last_error.code == "tool_use_failed"
            if retryable and attempt < self.max_retries:
                wait = _retry_after(response, default=2 * (attempt + 1))
                logger.warning("LLM HTTP %s, nova tentativa em %ss", response.status_code, wait)
                time.sleep(wait)
                continue
            raise last_error

        assert last_error is not None
        raise last_error

    def _http_error(self, response: requests.Response) -> LLMError:
        status = response.status_code
        detail = _extract_error_message(response)
        label = self.provider.label

        if status == 400 and "tool_use_failed" in detail:
            return LLMError(
                "O modelo gerou uma chamada de ferramenta malformada. Tente novamente ou troque de modelo.",
                detail=detail,
                code="tool_use_failed",
            )
        messages = {
            400: (
                f"O provedor {label} recusou a requisição. Verifique se o modelo '{self.model}' "
                "existe e suporta chamadas de ferramentas (function calling)."
            ),
            401: f"Chave de API inválida para o provedor {label}.",
            403: f"A chave de API não tem permissão para usar o modelo '{self.model}' no {label}.",
            404: (
                f"Modelo '{self.model}' não encontrado no provedor {label}. "
                "Confira o nome em GROQ_MODEL / GEMINI_MODEL no arquivo .env."
            ),
            413: "A conversa ficou grande demais para o modelo. Limpe o histórico e tente de novo.",
            429: (
                f"Limite de uso gratuito do {label} atingido (o plano gratuito tem limites por minuto e por dia)."
            ),
        }
        if status >= 500:
            message = f"O provedor {label} está instável no momento (HTTP {status}). Tente novamente."
        else:
            message = messages.get(status, f"Erro HTTP {status} ao chamar o modelo de IA.")
        return LLMError(message, detail=detail, code=f"http{status}")


def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(data, list) and data:  # Gemini às vezes devolve uma lista
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            parts = [str(err.get(k)) for k in ("code", "type", "message") if err.get(k)]
            if parts:
                return " | ".join(parts)[:500]
        return str(err)[:500]
    return str(data)[:500]


def _retry_after(response: requests.Response, default: float) -> float:
    try:
        return min(float(response.headers.get("Retry-After", default)), 10.0)
    except (TypeError, ValueError):
        return default
