import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
def update_multi_precision(self, index, weight, grad, state):
    use_multi_precision = self.multi_precision and weight.dtype == numpy.float16 and isinstance(state, (tuple, list))
    self._update_impl(index, weight, grad, state, multi_precision=use_multi_precision)