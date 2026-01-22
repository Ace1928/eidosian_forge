from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def is_grid_on():
    xticks = mpl.pyplot.gca().xaxis.get_major_ticks()
    yticks = mpl.pyplot.gca().yaxis.get_major_ticks()
    xoff = all((not g.gridline.get_visible() for g in xticks))
    yoff = all((not g.gridline.get_visible() for g in yticks))
    return not (xoff and yoff)