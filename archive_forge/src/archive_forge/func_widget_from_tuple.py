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
def widget_from_tuple(o, name, default=empty):
    """Make widgets from a tuple abbreviation."""
    int_default = default is empty or isinstance(default, int)
    if _matches(o, (Real, Real)):
        min, max, value = _get_min_max_value(o[0], o[1])
        if all((isinstance(_, Integral) for _ in o)) and int_default:
            cls = IntSlider
        else:
            cls = FloatSlider
        return cls(value=value, start=min, end=max, name=name)
    elif _matches(o, (Real, Real, Real)):
        step = o[2]
        if step <= 0:
            raise ValueError('step must be >= 0, not %r' % step)
        min, max, value = _get_min_max_value(o[0], o[1], step=step)
        if all((isinstance(_, Integral) for _ in o)) and int_default:
            cls = IntSlider
        else:
            cls = FloatSlider
        return cls(value=value, start=min, end=max, step=step, name=name)
    elif _matches(o, (Real, Real, Real, Real)):
        step = o[2]
        if step <= 0:
            raise ValueError('step must be >= 0, not %r' % step)
        min, max, value = _get_min_max_value(o[0], o[1], value=o[3], step=step)
        if all((isinstance(_, Integral) for _ in o)):
            cls = IntSlider
        else:
            cls = FloatSlider
        return cls(value=value, start=min, end=max, step=step, name=name)
    elif len(o) == 4:
        min, max, value = _get_min_max_value(o[0], o[1], value=o[3])
        if all((isinstance(_, Integral) for _ in [o[0], o[1], o[3]])):
            cls = IntSlider
        else:
            cls = FloatSlider
        return cls(value=value, start=min, end=max, name=name)