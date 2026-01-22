from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.utils import get_cwidth
def draw_all_floats(self) -> None:
    """
        Draw all float functions in order of z-index.
        """
    while self._draw_float_functions:
        functions = sorted(self._draw_float_functions, key=lambda item: item[0])
        self._draw_float_functions = functions[1:]
        functions[0][1]()