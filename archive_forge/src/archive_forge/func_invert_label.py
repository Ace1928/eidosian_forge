import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def invert_label(args, column):
    """Invert mapping.
    Find key corresponding to value column in dict args["labels"].
    Returns `column` if the value does not exist.
    """
    reversed_labels = {value: key for key, value in args['labels'].items()}
    try:
        return reversed_labels[column]
    except Exception:
        return column