from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def theme_update(**kwargs: themeable):
    """
    Modify elements of the current theme

    Parameters
    ----------
    kwargs : dict
        Theme elements
    """
    assert 'complete' not in kwargs
    theme_set(theme_get() + theme(**kwargs))