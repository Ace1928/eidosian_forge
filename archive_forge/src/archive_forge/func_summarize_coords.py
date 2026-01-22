from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def summarize_coords(variables) -> str:
    li_items = []
    for k, v in variables.items():
        li_content = summarize_variable(k, v, is_index=k in variables.xindexes)
        li_items.append(f"<li class='xr-var-item'>{li_content}</li>")
    vars_li = ''.join(li_items)
    return f"<ul class='xr-var-list'>{vars_li}</ul>"