from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def infer_map_type(self, norm, sizes, var_type):
    if norm is not None:
        map_type = 'numeric'
    elif isinstance(sizes, (dict, list)):
        map_type = 'categorical'
    else:
        map_type = var_type
    return map_type