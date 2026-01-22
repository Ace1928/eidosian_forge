from __future__ import annotations
from typing import (
import warnings
from matplotlib.artist import setp
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
import pandas as pd
import pandas.core.common as com
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def maybe_color_bp(self, bp) -> None:
    if isinstance(self.color, dict):
        boxes = self.color.get('boxes', self._boxes_c)
        whiskers = self.color.get('whiskers', self._whiskers_c)
        medians = self.color.get('medians', self._medians_c)
        caps = self.color.get('caps', self._caps_c)
    else:
        boxes = self.color or self._boxes_c
        whiskers = self.color or self._whiskers_c
        medians = self.color or self._medians_c
        caps = self.color or self._caps_c
    color_tup = (boxes, whiskers, medians, caps)
    maybe_color_bp(bp, color_tup=color_tup, **self.kwds)