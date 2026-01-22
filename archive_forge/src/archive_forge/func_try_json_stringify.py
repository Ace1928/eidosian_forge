import json
from typing import Any, Callable, List
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text
def try_json_stringify(obj: Any, fallback: str) -> str:
    """
    Try to stringify an object to JSON.
    Args:
        obj: Object to stringify.
        fallback: Fallback string to return if the object cannot be stringified.

    Returns:
        A JSON string if the object can be stringified, otherwise the fallback string.

    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return fallback