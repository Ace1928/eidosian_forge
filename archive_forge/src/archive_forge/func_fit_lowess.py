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
def fit_lowess(self):
    """Fit a locally-weighted regression, which returns its own grid."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    grid, yhat = lowess(self.y, self.x).T
    return (grid, yhat)