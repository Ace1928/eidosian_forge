from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def could_overlap(self, xyr_i, swarm):
    """Return a list of all swarm points that could overlap with target."""
    _, y_i, r_i = xyr_i
    neighbors = []
    for xyr_j in reversed(swarm):
        _, y_j, r_j = xyr_j
        if y_i - y_j < r_i + r_j:
            neighbors.append(xyr_j)
        else:
            break
    return np.array(neighbors)[::-1]