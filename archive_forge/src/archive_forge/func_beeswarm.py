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
def beeswarm(self, orig_xyr):
    """Adjust x position of points to avoid overlaps."""
    midline = orig_xyr[0, 0]
    swarm = np.atleast_2d(orig_xyr[0])
    for xyr_i in orig_xyr[1:]:
        neighbors = self.could_overlap(xyr_i, swarm)
        candidates = self.position_candidates(xyr_i, neighbors)
        offsets = np.abs(candidates[:, 0] - midline)
        candidates = candidates[np.argsort(offsets)]
        new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)
        swarm = np.vstack([swarm, new_xyr_i])
    return swarm