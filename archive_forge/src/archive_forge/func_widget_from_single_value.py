from __future__ import annotations
from collections.abc import Iterable, Mapping
from inspect import Parameter
from numbers import Integral, Number, Real
from typing import Any, Optional, Tuple
import param
from .base import Widget
from .input import Checkbox, TextInput
from .select import Select
from .slider import DiscreteSlider, FloatSlider, IntSlider
@staticmethod
def widget_from_single_value(o, name):
    """Make widgets from single values, which can be used as parameter defaults."""
    if isinstance(o, str):
        return TextInput(value=str(o), name=name)
    elif isinstance(o, bool):
        return Checkbox(value=o, name=name)
    elif isinstance(o, Integral):
        min, max, value = _get_min_max_value(None, None, o)
        return IntSlider(value=o, start=min, end=max, name=name)
    elif isinstance(o, Real):
        min, max, value = _get_min_max_value(None, None, o)
        return FloatSlider(value=o, start=min, end=max, name=name)
    else:
        return None