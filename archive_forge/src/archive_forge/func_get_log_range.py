import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def get_log_range(lo, hi):
    lo = np.floor(np.log(lo) / np.log(base))
    hi = np.ceil(np.log(hi) / np.log(base))
    return (lo, hi)