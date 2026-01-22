import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def named_colorscales():
    """
    Returns lowercased names of built-in continuous colorscales.
    """
    from _plotly_utils.basevalidators import ColorscaleValidator
    return [c for c in ColorscaleValidator('', '').named_colorscales]