"""Processamento do retorno da NewsAPI.

Transforma o JSON bruto em uma lista limpa de artigos:
- remove artigos removidos ("[Removed]") ou sem título/URL;
- elimina duplicatas (mesma URL ou mesmo título);
- limpa HTML e o sufixo "[+1234 chars]" que a NewsAPI adiciona;
- formata datas e trunca textos longos (economiza tokens do LLM);
- numera os artigos para que a IA possa citá-los como [1], [2]...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

_TRUNCATION_SUFFIX = re.compile(r"\s*\[\+\d+ chars\]\s*$")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_NON_WORD = re.compile(r"\W+")


@dataclass
class Article:
    id: int
    title: str
    source: str
    url: str
    published_at: str  # já formatada (dd/mm/aaaa HH:MM UTC)
    author: str = ""
    description: str = ""
    content: str = ""
    image_url: str = ""

    def to_llm_dict(self) -> dict[str, Any]:
        """Versão compacta enviada ao modelo de linguagem."""
        data = {
            "id": self.id,
            "titulo": self.title,
            "fonte": self.source,
            "data": self.published_at,
            "descricao": self.description,
        }
        if self.content and self.content != self.description:
            data["trecho"] = self.content
        return data


def clean_text(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    text = _TRUNCATION_SUFFIX.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if len(text) > max_len:
        cut = text[:max_len].rsplit(" ", 1)[0]
        text = cut.rstrip(".,;:") + "…"
    return text


def format_date(iso_date: str | None) -> str:
    if not iso_date:
        return ""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    except ValueError:
        return iso_date
    return dt.strftime("%d/%m/%Y %H:%M UTC")


def _normalize_title(title: str) -> str:
    return _NON_WORD.sub(" ", title.lower()).strip()


def process_articles(
    raw_articles: Iterable[dict[str, Any]] | None,
    start_id: int = 1,
    limit: int = 10,
    known_urls: set[str] | None = None,
) -> list[Article]:
    """Limpa, deduplica e numera os artigos retornados pela NewsAPI."""
    seen_urls = set(known_urls or ())
    seen_titles: set[str] = set()
    result: list[Article] = []

    for raw in raw_articles or []:
        if not isinstance(raw, dict):
            continue
        title = clean_text(raw.get("title"), 300)
        url = (raw.get("url") or "").strip()
        if not title or not url or title == "[Removed]" or "removed.com" in url:
            continue

        norm_title = _normalize_title(title)
        if url in seen_urls or norm_title in seen_titles:
            continue
        seen_urls.add(url)
        seen_titles.add(norm_title)

        source = (raw.get("source") or {}).get("name") or "Fonte desconhecida"
        result.append(
            Article(
                id=start_id + len(result),
                title=title,
                source=source.strip(),
                url=url,
                published_at=format_date(raw.get("publishedAt")),
                author=clean_text(raw.get("author"), 120),
                description=clean_text(raw.get("description"), 400),
                content=clean_text(raw.get("content"), 500),
                image_url=(raw.get("urlToImage") or "").strip(),
            )
        )
        if len(result) >= limit:
            break

    return result


def process_sources(raw_sources: Iterable[dict[str, Any]] | None, limit: int = 40) -> list[dict[str, str]]:
    """Simplifica a lista de veículos retornada por /top-headlines/sources."""
    result = []
    for raw in raw_sources or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        result.append(
            {
                "id": raw.get("id") or "",
                "nome": raw.get("name", ""),
                "descricao": clean_text(raw.get("description"), 160),
                "categoria": raw.get("category") or "",
                "idioma": raw.get("language") or "",
                "pais": raw.get("country") or "",
                "url": raw.get("url") or "",
            }
        )
        if len(result) >= limit:
            break
    return result
