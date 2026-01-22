from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
def property_link(obj: Any) -> str:
    return f':class:`~bokeh.core.properties.{obj.__class__.__name__}`\\ '