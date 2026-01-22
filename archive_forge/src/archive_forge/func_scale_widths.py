from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default
def scale_widths(w):
    empty = 0 if self.empty == 'fill' else w.mean()
    filled = w.fillna(empty)
    scale = filled.max()
    norm = filled.sum()
    if self.empty == 'keep':
        w = filled
    return w / norm * scale