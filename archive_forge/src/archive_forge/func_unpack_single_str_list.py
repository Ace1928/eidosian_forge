from __future__ import annotations
import random
from typing import TYPE_CHECKING
from matplotlib import patches
import matplotlib.lines as mlines
import numpy as np
from pandas.core.dtypes.missing import notna
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def unpack_single_str_list(keys):
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]
    return keys