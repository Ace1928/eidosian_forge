import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
Initialize kwargs dict.

    If the input is a dictionary, it returns
    a copy of the dictionary, otherwise it
    returns an empty dictionary.

    Parameters
    ----------
    kwargs : dict or None
        kwargs dict to initialize
    