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
def set_lr_mult(self, args_lr_mult):
    """Sets an individual learning rate multiplier for each parameter.

        If you specify a learning rate multiplier for a parameter, then
        the learning rate for the parameter will be set as the product of
        the global learning rate `self.lr` and its multiplier.

        .. note:: The default learning rate multiplier of a `Variable`
            can be set with `lr_mult` argument in the constructor.

        Parameters
        ----------
        args_lr_mult : dict of str/int to float
            For each of its key-value entries, the learning rate multipler for the
            parameter specified in the key will be set as the given value.

            You can specify the parameter with either its name or its index.
            If you use the name, you should pass `sym` in the constructor,
            and the name you specified in the key of `args_lr_mult` should match
            the name of the parameter in `sym`. If you use the index, it should
            correspond to the index of the parameter used in the `update` method.

            Specifying a parameter by its index is only supported for backward
            compatibility, and we recommend to use the name instead.
        """
    self.lr_mult = {}
    if self.sym_info:
        attr, arg_names = self.sym_info
        for name in arg_names:
            if name in attr and '__lr_mult__' in attr[name]:
                self.lr_mult[name] = float(attr[name]['__lr_mult__'])
    self.lr_mult.update(args_lr_mult)