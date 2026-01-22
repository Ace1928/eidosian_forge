from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def f_cb(x):
    f_cb.nfev += 1
    return np.sum(np.abs(self.f_cb(x, self.internal_params)))