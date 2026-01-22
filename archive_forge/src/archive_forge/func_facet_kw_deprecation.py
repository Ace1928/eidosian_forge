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
def facet_kw_deprecation(key, val):
    msg = f'{key} is deprecated from the `lmplot` function signature. Please update your code to pass it using `facet_kws`.'
    if val is not None:
        warnings.warn(msg, UserWarning)
        facet_kws[key] = val