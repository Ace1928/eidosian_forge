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
def vectorized_to_hex(c_values, keep_alpha=False):
    """Convert a color (including vector of colors) to hex.

    Parameters
    ----------
    c: Matplotlib color

    keep_alpha: boolean
        to select if alpha values should be kept in the final hex values.

    Returns
    -------
    rgba_hex : vector of hex values
    """
    try:
        hex_color = to_hex(c_values, keep_alpha)
    except ValueError:
        hex_color = [to_hex(color, keep_alpha) for color in c_values]
    return hex_color