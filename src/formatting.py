"""Funções de apresentação (independentes do Streamlit, para facilitar testes)."""
from __future__ import annotations

import re

from .processing import Article

_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\](?!\()")


def extract_citations(answer: str) -> list[int]:
    """Retorna os ids citados na resposta, na ordem em que aparecem."""
    ids: list[int] = []
    for match in _CITATION_RE.finditer(answer or ""):
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if n not in ids:
                ids.append(n)
    return ids


def link_citations(answer: str, articles: list[Article]) -> str:
    """Transforma [3] em um link clicável para a notícia 3."""
    by_id = {a.id: a for a in articles}

    def replace(match: re.Match[str]) -> str:
        parts = []
        for raw in re.split(r"[,;]", match.group(1)):
            n = int(raw.strip())
            article = by_id.get(n)
            parts.append(f"[[{n}]]({article.url} \"{_escape_title(article.source)}\")" if article else f"[{n}]")
        return "".join(parts)

    return _CITATION_RE.sub(replace, answer or "")


def escape_markdown_link_text(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")


def _escape_title(text: str) -> str:
    return text.replace('"', "'")
