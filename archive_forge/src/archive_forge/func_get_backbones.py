from . import r_function
import pandas as pd
def get_backbones():
    """Output full list of cell trajectory backbones.

    Returns
    -------
    backbones: array of backbone names
    """
    return _get_backbones()