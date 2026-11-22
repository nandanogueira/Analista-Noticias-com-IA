"""Ferramentas disponíveis para a IA (function calling).

A IA recebe as definições em ``TOOL_DEFINITIONS`` e decide sozinha qual
ferramenta chamar (e com quais argumentos) conforme a pergunta do usuário:

- ``buscar_noticias``      -> assunto específico (NewsAPI /everything)
- ``manchetes_principais`` -> "o que está acontecendo", destaques por país/categoria
- ``listar_fontes``        -> quais veículos estão disponíveis

O ``ToolExecutor`` valida os argumentos gerados pela IA, executa a consulta
real na NewsAPI, processa o retorno e devolve um JSON compacto para a IA.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

from .config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, NEWSAPI_MAX_DAYS_BACK
from .errors import NewsAPIError, ToolArgumentError
from .news_api import NewsAPIClient
from .processing import Article, process_articles, process_sources

logger = logging.getLogger(__name__)

VALID_LANGUAGES = {"ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv", "ud", "zh"}
VALID_SORT = {"publishedAt", "relevancy", "popularity"}
VALID_CATEGORIES = {"business", "entertainment", "general", "health", "science", "sports", "technology"}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "buscar_noticias",
            "description": (
                "Busca notícias recentes (até ~30 dias) sobre um ASSUNTO ESPECÍFICO por palavras-chave, "
                "em milhares de veículos. Use para perguntas sobre um tema, pessoa, empresa, evento ou lugar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "Palavras-chave de busca (não a pergunta inteira). Aceita operadores: "
                            'aspas para frase exata, AND, OR, NOT. Ex.: "inteligência artificial" AND Brasil'
                        ),
                    },
                    "idioma": {
                        "type": "string",
                        "enum": sorted(VALID_LANGUAGES) + ["todos"],
                        "description": "Idioma das notícias. Padrão 'pt'. Use 'todos' para não filtrar.",
                    },
                    "ordenar_por": {
                        "type": "string",
                        "enum": sorted(VALID_SORT),
                        "description": "publishedAt = mais recentes (padrão); relevancy = mais relevantes; popularity = mais populares.",
                    },
                    "data_inicial": {
                        "type": "string",
                        "description": "Data inicial no formato AAAA-MM-DD (opcional).",
                    },
                    "data_final": {
                        "type": "string",
                        "description": "Data final no formato AAAA-MM-DD (opcional).",
                    },
                    "dominios": {
                        "type": "string",
                        "description": "Restringe a domínios específicos, separados por vírgula. Ex.: g1.globo.com,folha.uol.com.br (opcional).",
                    },
                    "quantidade": {
                        "type": "integer",
                        "description": f"Quantas notícias trazer (1 a {MAX_PAGE_SIZE}). Padrão {DEFAULT_PAGE_SIZE}.",
                    },
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manchetes_principais",
            "description": (
                "Retorna as manchetes principais do momento de um país e/ou categoria. Use quando o usuário "
                "perguntar de forma geral 'o que está acontecendo', 'destaques de hoje', 'principais notícias de esportes' etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pais": {
                        "type": "string",
                        "description": "Código ISO de 2 letras do país (ex.: us, br, gb, pt). Padrão: us.",
                    },
                    "categoria": {
                        "type": "string",
                        "enum": sorted(VALID_CATEGORIES),
                        "description": "Categoria das manchetes (opcional).",
                    },
                    "consulta": {
                        "type": "string",
                        "description": "Palavra-chave para filtrar as manchetes (opcional).",
                    },
                    "quantidade": {
                        "type": "integer",
                        "description": f"Quantas manchetes trazer (1 a {MAX_PAGE_SIZE}). Padrão {DEFAULT_PAGE_SIZE}.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_fontes",
            "description": "Lista os veículos de notícias (fontes) disponíveis na NewsAPI, filtrando por idioma, país ou categoria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idioma": {"type": "string", "description": "Código do idioma (ex.: pt, en)."},
                    "pais": {"type": "string", "description": "Código ISO de 2 letras do país (ex.: br, us)."},
                    "categoria": {"type": "string", "enum": sorted(VALID_CATEGORIES)},
                },
                "required": [],
            },
        },
    },
]

TOOL_LABELS = {
    "buscar_noticias": "🔎 Buscar notícias por assunto",
    "manchetes_principais": "🗞️ Manchetes principais",
    "listar_fontes": "🏢 Listar fontes",
}


@dataclass
class ToolCallRecord:
    """Registro de uma chamada de ferramenta (exibido na interface)."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    api_params: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    summary: str = ""
    total_results: int = 0
    article_ids: list[int] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: int = 0


class ToolExecutor:
    def __init__(self, news_client: NewsAPIClient):
        self.news = news_client
        self.articles: list[Article] = []
        self.records: list[ToolCallRecord] = []
        self._handlers: dict[str, Callable[[dict[str, Any], ToolCallRecord], dict[str, Any]]] = {
            "buscar_noticias": self._buscar_noticias,
            "manchetes_principais": self._manchetes_principais,
            "listar_fontes": self._listar_fontes,
        }

    # ------------------------------------------------------------------ #
    def execute(self, name: str, raw_arguments: Any) -> str:
        """Executa a ferramenta e devolve o resultado (JSON) para a IA.

        Erros recuperáveis (argumento inválido, parâmetro recusado pela API)
        são devolvidos à IA para que ela ajuste a chamada. Erros fatais
        (chave inválida, limite atingido, sem conexão) são propagados.
        """
        record = ToolCallRecord(name=name)
        self.records.append(record)
        start = time.perf_counter()
        try:
            args = _parse_arguments(raw_arguments)
            record.arguments = args
            handler = self._handlers.get(name)
            if handler is None:
                raise ToolArgumentError(
                    f"Ferramenta desconhecida: '{name}'. Use uma de: {', '.join(self._handlers)}."
                )
            payload = handler(args, record)
            record.ok = True
        except ToolArgumentError as exc:
            record.error = exc.user_message
            payload = {"erro": exc.user_message}
        except NewsAPIError as exc:
            record.error = exc.user_message
            if exc.fatal:
                raise
            payload = {"erro": exc.user_message, "detalhe": exc.detail}
        finally:
            record.elapsed_ms = int((time.perf_counter() - start) * 1000)

        logger.info("Ferramenta %s(%s) -> %s", name, record.arguments, record.summary or record.error)
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Ferramentas
    # ------------------------------------------------------------------ #
    def _buscar_noticias(self, args: dict[str, Any], record: ToolCallRecord) -> dict[str, Any]:
        consulta = _require_str(args, "consulta", max_len=500)

        idioma = _optional_str(args, "idioma", default="pt").lower()
        if idioma in ("todos", "all", "*"):
            idioma = None
        elif idioma not in VALID_LANGUAGES:
            raise ToolArgumentError(f"Idioma inválido: '{idioma}'. Use um de: {', '.join(sorted(VALID_LANGUAGES))}.")

        ordenar = _optional_str(args, "ordenar_por", default="publishedAt")
        if ordenar not in VALID_SORT:
            ordenar = "publishedAt"

        data_inicial = _parse_date(args.get("data_inicial"), "data_inicial")
        data_final = _parse_date(args.get("data_final"), "data_final")
        limite = date.today() - timedelta(days=NEWSAPI_MAX_DAYS_BACK)
        if data_inicial and data_inicial < limite:
            record.warnings.append(
                f"O plano gratuito da NewsAPI só busca notícias do último mês; "
                f"data inicial ajustada de {data_inicial:%d/%m/%Y} para {limite:%d/%m/%Y}."
            )
            data_inicial = limite
        if data_inicial and data_final and data_final < data_inicial:
            raise ToolArgumentError("data_final não pode ser anterior a data_inicial.")

        dominios = _optional_str(args, "dominios", default="").replace(" ", "") or None
        quantidade = _page_size(args.get("quantidade"))

        record.api_params = {
            "endpoint": "/v2/everything",
            "q": consulta,
            "language": idioma,
            "sortBy": ordenar,
            "from": data_inicial.isoformat() if data_inicial else None,
            "to": data_final.isoformat() if data_final else None,
            "domains": dominios,
            "pageSize": quantidade,
        }
        data = self.news.everything(
            q=consulta,
            language=idioma,
            sort_by=ordenar,
            from_date=record.api_params["from"],
            to_date=record.api_params["to"],
            domains=dominios,
            page_size=quantidade,
        )
        return self._articles_payload(data, quantidade, record)

    def _manchetes_principais(self, args: dict[str, Any], record: ToolCallRecord) -> dict[str, Any]:
        pais = _optional_str(args, "pais", default="").lower() or None
        if pais and not re.fullmatch(r"[a-z]{2}", pais):
            raise ToolArgumentError(f"Código de país inválido: '{pais}'. Use 2 letras, ex.: br, us.")
        categoria = _optional_str(args, "categoria", default="").lower() or None
        if categoria and categoria not in VALID_CATEGORIES:
            raise ToolArgumentError(
                f"Categoria inválida: '{categoria}'. Use uma de: {', '.join(sorted(VALID_CATEGORIES))}."
            )
        consulta = _optional_str(args, "consulta", default="") or None
        if not (pais or categoria or consulta):
            pais = "us"  # a NewsAPI exige ao menos um filtro
        quantidade = _page_size(args.get("quantidade"))

        record.api_params = {
            "endpoint": "/v2/top-headlines",
            "country": pais,
            "category": categoria,
            "q": consulta,
            "pageSize": quantidade,
        }
        data = self.news.top_headlines(country=pais, category=categoria, q=consulta, page_size=quantidade)
        payload = self._articles_payload(data, quantidade, record)
        if not payload["artigos"]:
            payload["sugestao"] = (
                "Nenhuma manchete para esse filtro (a NewsAPI tem poucas manchetes para alguns países). "
                "Tente a ferramenta buscar_noticias com palavras-chave e idioma adequados."
            )
        return payload

    def _listar_fontes(self, args: dict[str, Any], record: ToolCallRecord) -> dict[str, Any]:
        idioma = _optional_str(args, "idioma", default="").lower() or None
        pais = _optional_str(args, "pais", default="").lower() or None
        categoria = _optional_str(args, "categoria", default="").lower() or None
        record.api_params = {
            "endpoint": "/v2/top-headlines/sources",
            "language": idioma,
            "country": pais,
            "category": categoria,
        }
        data = self.news.sources(language=idioma, country=pais, category=categoria)
        sources = process_sources(data.get("sources"))
        record.sources = sources
        record.total_results = len(data.get("sources") or [])
        record.summary = f"{len(sources)} fontes listadas"
        return {
            "total_fontes": record.total_results,
            "fontes": [
                {k: s[k] for k in ("nome", "categoria", "idioma", "pais", "url")} for s in sources
            ],
        }

    # ------------------------------------------------------------------ #
    def _articles_payload(self, data: dict[str, Any], limit: int, record: ToolCallRecord) -> dict[str, Any]:
        raw = data.get("articles") or []
        new_articles = process_articles(
            raw,
            start_id=len(self.articles) + 1,
            limit=limit,
            known_urls={a.url for a in self.articles},
        )
        self.articles.extend(new_articles)
        record.total_results = int(data.get("totalResults") or 0)
        record.article_ids = [a.id for a in new_articles]
        descartados = len(raw) - len(new_articles) if len(new_articles) < limit else 0
        record.summary = (
            f"{record.total_results} resultados na API · {len(new_articles)} notícias processadas"
            + (f" · {descartados} descartadas (duplicadas/removidas)" if descartados > 0 else "")
        )

        payload: dict[str, Any] = {
            "total_encontrado": record.total_results,
            "retornados": len(new_articles),
            "artigos": [a.to_llm_dict() for a in new_articles],
        }
        if record.warnings:
            payload["avisos"] = record.warnings
        if not new_articles:
            payload["mensagem"] = (
                "Nenhuma notícia encontrada. Considere termos mais amplos, sinônimos, "
                "outro idioma (ex.: 'en') ou remover filtros de data."
            )
        return payload


# ---------------------------------------------------------------------- #
# Validação de argumentos gerados pela IA
# ---------------------------------------------------------------------- #
def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(f"Os argumentos da ferramenta não são um JSON válido: {exc.msg}.") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ToolArgumentError("Os argumentos da ferramenta devem ser um objeto JSON.")


def _require_str(args: dict[str, Any], key: str, max_len: int) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentError(f"O argumento '{key}' é obrigatório e deve ser um texto.")
    value = value.strip()
    if len(value) > max_len:
        raise ToolArgumentError(f"O argumento '{key}' excede {max_len} caracteres.")
    return value


def _optional_str(args: dict[str, Any], key: str, default: str) -> str:
    value = args.get(key)
    if value is None:
        return default
    return str(value).strip() or default


def _parse_date(value: Any, key: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ToolArgumentError(f"'{key}' deve estar no formato AAAA-MM-DD (recebido: {value}).") from exc


def _page_size(value: Any) -> int:
    try:
        size = int(value) if value not in (None, "") else DEFAULT_PAGE_SIZE
    except (TypeError, ValueError):
        size = DEFAULT_PAGE_SIZE
    return max(1, min(size, MAX_PAGE_SIZE))
