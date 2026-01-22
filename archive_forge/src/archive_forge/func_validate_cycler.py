import ast
from functools import lru_cache, reduce
from numbers import Real
import operator
import os
import re
import numpy as np
from matplotlib import _api, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap, is_color_like
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
from matplotlib._enums import JoinStyle, CapStyle
from cycler import Cycler, cycler as ccycler
def validate_cycler(s):
    """Return a Cycler object from a string repr or the object itself."""
    if isinstance(s, str):
        try:
            _DunderChecker().visit(ast.parse(s))
            s = eval(s, {'cycler': cycler, '__builtins__': {}})
        except BaseException as e:
            raise ValueError(f'{s!r} is not a valid cycler construction: {e}') from e
    if isinstance(s, Cycler):
        cycler_inst = s
    else:
        raise ValueError(f'Object is not a string or Cycler instance: {s!r}')
    unknowns = cycler_inst.keys - (set(_prop_validators) | set(_prop_aliases))
    if unknowns:
        raise ValueError('Unknown artist properties: %s' % unknowns)
    checker = set()
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        if norm_prop != prop and norm_prop in cycler_inst.keys:
            raise ValueError(f'Cannot specify both {norm_prop!r} and alias {prop!r} in the same prop_cycle')
        if norm_prop in checker:
            raise ValueError(f'Another property was already aliased to {norm_prop!r}. Collision normalizing {prop!r}.')
        checker.update([norm_prop])
    assert len(checker) == len(cycler_inst.keys)
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        cycler_inst.change_key(prop, norm_prop)
    for key, vals in cycler_inst.by_key().items():
        _prop_validators[key](vals)
    return cycler_inst