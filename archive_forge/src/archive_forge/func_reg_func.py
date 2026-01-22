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
def reg_func(_x, _y):
    _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
    return np.linalg.pinv(_x).dot(_y)