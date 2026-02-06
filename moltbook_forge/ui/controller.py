#!/usr/bin/env python3
"""Controller helpers for Moltbook Nexus UI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def build_filter_maps(
    filters: dict,
    selected: str,
    page: int,
    page_size: int,
    urlencode,
) -> tuple[list[dict], callable, callable, str]:
    active_filters = []
    for key, value in filters.items():
        if value != "":
            active_filters.append({"key": key, "value": value})

    def _remove_filter_url(remove_key: str) -> str:
        remaining = {k: v for k, v in filters.items() if k != remove_key and v != ""}
        if selected:
            remaining["selected"] = selected
        if not remaining:
            return "/"
        return "/?" + urlencode(remaining)

    def _page_url(target_page: int) -> str:
        params = {k: v for k, v in filters.items() if v != ""}
        if selected:
            params["selected"] = selected
        params["page"] = target_page
        params["page_size"] = page_size
        return "/?" + urlencode(params)

    detail_params = {k: v for k, v in filters.items() if v != ""}
    if detail_params:
        detail_query_prefix = "/?" + urlencode(detail_params) + "&selected="
    else:
        detail_query_prefix = "/?selected="
    return active_filters, _remove_filter_url, _page_url, detail_query_prefix


def pack_evidence_summary(evidence_summary: dict) -> dict:
    return {
        post_id: {
            "lowest_cred": summary.lowest_cred,
            "url_count": summary.url_count,
        }
        for post_id, summary in evidence_summary.items()
    }
