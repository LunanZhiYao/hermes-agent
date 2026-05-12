#!/usr/bin/env python3
"""Alibaba Cloud AI Search Platform (OpenSearch) web search as a standalone tool.

This is **not** wired into ``web.backend`` / ``web_search``. Add the toolset
``aliyun_opensearch`` under ``platform_toolsets`` for your platform (e.g. ``cli``)
in ``~/.hermes/config.yaml``, alongside your usual toolsets.

Environment (see Alibaba docs for ops-web-search-001)::

    ALIYUN_OPENSEARCH_WEB_SEARCH_URL  — full POST endpoint URL
    ALIYUN_OPENSEARCH_API_KEY         — Bearer API key

Optional::

    ALIYUN_OPENSEARCH_CONTENT_TYPE    — summary | snippet (default summary)
    ALIYUN_OPENSEARCH_QUERY_REWRITE   — 1/true/yes for query rewrite (default on)

Docs: https://help.aliyun.com/zh/open-search/search-platform/developer-reference/web-search
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _aliyun_opensearch_configured() -> bool:
    url = (os.getenv("ALIYUN_OPENSEARCH_WEB_SEARCH_URL") or "").strip()
    key = (os.getenv("ALIYUN_OPENSEARCH_API_KEY") or "").strip()
    return bool(url and key)


def aliyun_opensearch_search(query: str, limit: int = 5) -> str:
    """Run web search; returns JSON like ``web_search`` (success + data.web[])."""
    from tools.interrupt import is_interrupted

    if is_interrupted():
        return json.dumps({"error": "Interrupted", "success": False}, ensure_ascii=False)

    q = (query or "").strip()
    if not q:
        return tool_error("query is required")

    api_url = (os.getenv("ALIYUN_OPENSEARCH_WEB_SEARCH_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("ALIYUN_OPENSEARCH_API_KEY") or "").strip()
    if not api_url or not api_key:
        return tool_error(
            "Aliyun OpenSearch requires ALIYUN_OPENSEARCH_WEB_SEARCH_URL and ALIYUN_OPENSEARCH_API_KEY "
            "(see https://help.aliyun.com/zh/open-search/search-platform/developer-reference/web-search)"
        )

    content_type = (os.getenv("ALIYUN_OPENSEARCH_CONTENT_TYPE") or "summary").strip().lower()
    if content_type not in ("snippet", "summary"):
        content_type = "summary"

    top_k = min(max(int(limit) if limit else 5, 1), 50)
    payload = {
        "history": [],
        "query": q,
        "query_rewrite": _env_bool("ALIYUN_OPENSEARCH_QUERY_REWRITE", True),
        "top_k": top_k,
        "content_type": content_type,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    logger.info(
        "Aliyun OpenSearch web search: %r (top_k=%s, content_type=%s)",
        q,
        top_k,
        content_type,
    )
    try:
        response = httpx.post(api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.debug("Aliyun OpenSearch request failed: %s", e)
        return tool_error(f"Aliyun OpenSearch request failed: {type(e).__name__}: {e}")

    if isinstance(data, dict) and data.get("code"):
        msg = data.get("message") or str(data.get("code"))
        return tool_error(f"Aliyun OpenSearch API error: {msg}")

    result = data.get("result") if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        result = {}
    raw_results = result.get("search_result") or []

    web_results: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("tilte") or ""
        link = item.get("link") or ""
        desc = (item.get("snippet") or item.get("content") or "").strip()
        pos = item.get("position")
        if not isinstance(pos, int):
            pos = i + 1
        web_results.append({
            "title": title,
            "url": link,
            "description": desc,
            "position": pos,
        })

    out = {"success": True, "data": {"web": web_results}}
    return json.dumps(out, indent=2, ensure_ascii=False)


ALIYUN_OPENSEARCH_SEARCH_SCHEMA = {
    "name": "aliyun_opensearch_search",
    "description": (
        "Search the web using Alibaba Cloud AI Search Platform (OpenSearch web-search API). "
        "Returns titles, URLs, and snippets/summaries. Search-only — use web_extract or browser "
        "tools for page content. Same response shape as web_search (data.web[])."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (backend-supported operators may apply).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (1–50). Default 5.",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

registry.register(
    name="aliyun_opensearch",
    toolset="aliyun_opensearch",
    schema=ALIYUN_OPENSEARCH_SEARCH_SCHEMA,
    handler=lambda args, **kw: aliyun_opensearch_search(
        args.get("query", ""),
        limit=args.get("limit", 5),
    ),
    check_fn=_aliyun_opensearch_configured,
    requires_env=["ALIYUN_OPENSEARCH_WEB_SEARCH_URL", "ALIYUN_OPENSEARCH_API_KEY"],
    emoji="🔍",
    max_result_size_chars=100_000,
)
