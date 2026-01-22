from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def css_bar(start: float, end: float, color: str) -> str:
    """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.

        Parameters
        ----------
        start : float
            Relative positional start of bar coloring in [0,1]
        end : float
            Relative positional end of the bar coloring in [0,1]
        color : str
            CSS valid color to apply.

        Returns
        -------
        str : The CSS applicable to the cell.

        Notes
        -----
        Uses ``base_css`` from outer scope.
        """
    cell_css = base_css
    if end > start:
        cell_css += 'background: linear-gradient(90deg,'
        if start > 0:
            cell_css += f' transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,'
        cell_css += f' {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)'
    return cell_css