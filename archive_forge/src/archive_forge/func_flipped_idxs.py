from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def flipped_idxs(ndim, axes):
    new_idxs = [slice(None)] * ndim
    for ax in axes:
        new_idxs[ax] = slice(None, None, -1)
    return tuple(new_idxs)