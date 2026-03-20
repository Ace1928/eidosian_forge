from __future__ import annotations

import json
from typing import Iterable


def coerce_string_list(value: object, *, lowercase: bool = False) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                value = parsed
            else:
                value = text
        if isinstance(value, str):
            separators = [",", "\n", ";"]
            parts = [value]
            for separator in separators:
                if separator in value:
                    parts = [part for chunk in parts for part in chunk.split(separator)]
            items = [part.strip() for part in parts if part and part.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [str(value).strip()]
    if lowercase:
        items = [item.lower() for item in items]
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def coerce_tag_list(value: object) -> list[str]:
    return coerce_string_list(value, lowercase=False)


def coerce_extension_list(value: object) -> list[str]:
    coerced = coerce_string_list(value, lowercase=True)
    return [item.lstrip(".") for item in coerced]
