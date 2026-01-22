from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def short_data_repr_html(array) -> str:
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, 'variable', array)._data
    if hasattr(internal_data, '_repr_html_'):
        return internal_data._repr_html_()
    text = escape(short_data_repr(array))
    return f'<pre>{text}</pre>'