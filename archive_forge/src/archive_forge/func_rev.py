import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def rev(x_mod):
    i = np.flatnonzero(i_f)
    N = len(i)
    index_offset = np.arange(N)
    insert_indices = i - index_offset
    x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
    return x_rev