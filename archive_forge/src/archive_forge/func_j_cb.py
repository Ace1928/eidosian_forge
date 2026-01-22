from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def j_cb(x):
    j_cb.njev += 1
    return self.j_cb(x, self.internal_params)