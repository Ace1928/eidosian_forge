import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def dot_generalized(a, b):
    a = asarray(a)
    if a.ndim >= 3:
        if a.ndim == b.ndim:
            new_shape = a.shape[:-1] + b.shape[-1:]
        elif a.ndim == b.ndim + 1:
            new_shape = a.shape[:-1]
        else:
            raise ValueError('Not implemented...')
        r = np.empty(new_shape, dtype=np.common_type(a, b))
        for c in itertools.product(*map(range, a.shape[:-2])):
            r[c] = dot(a[c], b[c])
        return r
    else:
        return dot(a, b)