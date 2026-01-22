import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
@property
def scatter_data(self):
    """Data where each observation is a point."""
    x_j = self.x_jitter
    if x_j is None:
        x = self.x
    else:
        x = self.x + np.random.uniform(-x_j, x_j, len(self.x))
    y_j = self.y_jitter
    if y_j is None:
        y = self.y
    else:
        y = self.y + np.random.uniform(-y_j, y_j, len(self.y))
    return (x, y)