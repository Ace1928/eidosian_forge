from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
def type_link(obj: Any) -> str:
    return _type_links.get(obj.__class__, property_link)(obj)