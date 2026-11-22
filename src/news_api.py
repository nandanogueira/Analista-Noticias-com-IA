"""Cliente HTTP da NewsAPI (https://newsapi.org/docs) usando ``requests``.

Endpoints usados:
- ``/v2/everything``              -> busca por palavras-chave (idioma, período, domínios)
- ``/v2/top-headlines``           -> manchetes principais por país/categoria
- ``/v2/top-headlines/sources``   -> veículos (fontes) disponíveis
"""
from __future__ import annotations

from typing import Any

import requests

from .config import NEWSAPI_TIMEOUT
from .errors import ConfigError, NewsAPIError

BASE_URL = "https://newsapi.org/v2"

# Códigos de erro documentados em https://newsapi.org/docs/errors
_ERROR_MESSAGES = {
    "apiKeyMissing": "A chave da NewsAPI não foi enviada. Configure NEWSAPI_KEY.",
    "apiKeyInvalid": "A chave da NewsAPI é inválida. Confira o valor de NEWSAPI_KEY.",
    "apiKeyDisabled": "A chave da NewsAPI foi desativada.",
    "apiKeyExhausted": "A chave da NewsAPI não tem mais requisições disponíveis.",
    "rateLimited": (
        "Limite de requisições da NewsAPI atingido (plano gratuito: 100 por dia). "
        "Tente novamente mais tarde."
    ),
    "parameterInvalid": "A consulta enviada à NewsAPI tem um parâmetro inválido.",
    "parametersMissing": "Faltam parâmetros obrigatórios na consulta à NewsAPI.",
    "sourcesTooMany": "Foram informadas fontes demais (máximo 20).",
    "sourceDoesNotExist": "Uma das fontes informadas não existe na NewsAPI.",
    "maximumResultsReached": "O plano gratuito da NewsAPI só permite acessar os 100 primeiros resultados.",
    "unexpectedError": "A NewsAPI apresentou um erro inesperado.",
}


class NewsAPIClient:
    def __init__(
        self,
        api_key: str | None,
        timeout: int = NEWSAPI_TIMEOUT,
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise ConfigError(
                "Chave da NewsAPI não configurada. Defina NEWSAPI_KEY no arquivo .env "
                "e reinicie a aplicação. Crie uma grátis em https://newsapi.org/register"
            )
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def everything(
        self,
        q: str,
        language: str | None = None,
        sort_by: str = "publishedAt",
        from_date: str | None = None,
        to_date: str | None = None,
        domains: str | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        return self._get(
            "everything",
            {
                "q": q,
                "language": language,
                "sortBy": sort_by,
                "from": from_date,
                "to": to_date,
                "domains": domains,
                "pageSize": page_size,
            },
        )

    def top_headlines(
        self,
        country: str | None = None,
        category: str | None = None,
        q: str | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        return self._get(
            "top-headlines",
            {"country": country, "category": category, "q": q, "pageSize": page_size},
        )

    def sources(
        self,
        language: str | None = None,
        country: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "top-headlines/sources",
            {"language": language, "country": country, "category": category},
        )

    # ------------------------------------------------------------------ #
    # Requisição + tratamento de erros
    # ------------------------------------------------------------------ #
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        clean_params = {k: v for k, v in params.items() if v not in (None, "", [])}
        url = f"{BASE_URL}/{endpoint}"
        try:
            response = self.session.get(
                url,
                params=clean_params,
                headers={"X-Api-Key": self.api_key, "User-Agent": "analista-de-noticias/1.0"},
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise NewsAPIError(
                "A NewsAPI demorou demais para responder. Tente novamente em instantes.",
                detail=str(exc),
                code="timeout",
            ) from exc
        except requests.ConnectionError as exc:
            raise NewsAPIError(
                "Não foi possível conectar à NewsAPI. Verifique sua conexão com a internet.",
                detail=str(exc),
                code="connection",
            ) from exc
        except requests.RequestException as exc:
            raise NewsAPIError("Falha inesperada ao consultar a NewsAPI.", detail=str(exc)) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise NewsAPIError(
                f"A NewsAPI retornou uma resposta inválida (HTTP {response.status_code}).",
                detail=response.text[:300],
            ) from exc

        if response.status_code != 200 or data.get("status") != "ok":
            code = data.get("code") or f"http{response.status_code}"
            message = _ERROR_MESSAGES.get(code, "A NewsAPI recusou a consulta.")
            raise NewsAPIError(message, detail=f"{code}: {data.get('message', '')}", code=code)

        return data
