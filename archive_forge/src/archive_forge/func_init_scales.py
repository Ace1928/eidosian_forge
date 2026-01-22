from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def init_scales(self, layout: pd.DataFrame, x_scale: Optional[scale]=None, y_scale: Optional[scale]=None) -> types.SimpleNamespace:
    scales = types.SimpleNamespace()
    if x_scale is not None:
        n = layout['SCALE_X'].max()
        scales.x = Scales([x_scale.clone() for i in range(n)])
    if y_scale is not None:
        n = layout['SCALE_Y'].max()
        scales.y = Scales([y_scale.clone() for i in range(n)])
    return scales