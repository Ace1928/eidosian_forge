from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from seaborn.utils import _version_predates
def share_axis(ax0, ax1, which):
    """Handle changes to post-hoc axis sharing."""
    if _version_predates(mpl, '3.5'):
        group = getattr(ax0, f'get_shared_{which}_axes')()
        group.join(ax1, ax0)
    else:
        getattr(ax1, f'share{which}')(ax0)