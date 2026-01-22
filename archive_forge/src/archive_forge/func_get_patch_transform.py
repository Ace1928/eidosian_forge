from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def get_patch_transform(self):
    if self._patch_type in ('arc', 'circle'):
        self._recompute_transform()
        return self._patch_transform
    else:
        return super().get_patch_transform()