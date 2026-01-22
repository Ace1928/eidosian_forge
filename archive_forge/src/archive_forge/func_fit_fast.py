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
def fit_fast(self, grid):
    """Low-level regression and prediction using linear algebra."""

    def reg_func(_x, _y):
        return np.linalg.pinv(_x).dot(_y)
    X, y = (np.c_[np.ones(len(self.x)), self.x], self.y)
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(X, y))
    if self.ci is None:
        return (yhat, None)
    beta_boots = algo.bootstrap(X, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed).T
    yhat_boots = grid.dot(beta_boots).T
    return (yhat, yhat_boots)