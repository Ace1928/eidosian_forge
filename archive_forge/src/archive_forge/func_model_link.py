from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
def model_link(fullname: str) -> str:
    return f':class:`~{fullname}`\\ '