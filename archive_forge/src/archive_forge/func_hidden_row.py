from __future__ import annotations
import logging # isort:skip
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from ..util.serialization import make_id
from ..util.strings import append_docstring
def hidden_row(c: str):
    return f'<div class="{cls_name}" style="display: none;">{c}</div>'