from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from seaborn.utils import _version_predates
def groupby_apply_include_groups(val):
    if _version_predates(pd, '2.2.0'):
        return {}
    return {'include_groups': val}