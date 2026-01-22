import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def get_rotate_label(self, text):
    if self._rotate_label is not None:
        return self._rotate_label
    else:
        return len(text) > 4