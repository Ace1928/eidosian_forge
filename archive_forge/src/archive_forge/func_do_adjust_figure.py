from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def do_adjust_figure(fig: Figure) -> bool:
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, 'get_constrained_layout'):
        return False
    return not fig.get_constrained_layout()