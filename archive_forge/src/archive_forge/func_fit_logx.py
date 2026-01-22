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
def fit_logx(self, grid):
    """Fit the model in log-space."""
    X, y = (np.c_[np.ones(len(self.x)), self.x], self.y)
    grid = np.c_[np.ones(len(grid)), np.log(grid)]

    def reg_func(_x, _y):
        _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
        return np.linalg.pinv(_x).dot(_y)
    yhat = grid.dot(reg_func(X, y))
    if self.ci is None:
        return (yhat, None)
    beta_boots = algo.bootstrap(X, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed).T
    yhat_boots = grid.dot(beta_boots).T
    return (yhat, yhat_boots)