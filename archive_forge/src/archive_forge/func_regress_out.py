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
def regress_out(self, a, b):
    """Regress b from a keeping a's original mean."""
    a_mean = a.mean()
    a = a - a_mean
    b = b - b.mean()
    b = np.c_[b]
    a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
    return np.asarray(a_prime + a_mean).reshape(a.shape)