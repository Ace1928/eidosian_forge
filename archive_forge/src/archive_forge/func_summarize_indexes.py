from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def summarize_indexes(indexes) -> str:
    indexes_li = ''.join((f"<li class='xr-var-item'>{summarize_index(v, i)}</li>" for v, i in indexes.items()))
    return f"<ul class='xr-var-list'>{indexes_li}</ul>"