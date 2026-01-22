import itertools
import functools
import importlib.util
def get_default_colors_dict(colors):
    import numpy as np
    colors = dict() if colors is None else dict(colors)
    colors.setdefault('None', np.array([0.5, 0.5, 0.5]))
    colors.setdefault('getitem', np.array([0.5, 0.5, 0.5]))
    return colors