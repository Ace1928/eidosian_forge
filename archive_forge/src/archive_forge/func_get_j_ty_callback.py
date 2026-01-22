from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_j_ty_callback(self):
    """ Generates a callback for evaluating the jacobian. """
    j_exprs = self.get_jac()
    if j_exprs is False:
        return None
    cb = self._callback_factory(j_exprs)
    if self.sparse:
        from scipy.sparse import csc_matrix

        def sparse_cb(x, y, p=()):
            data = cb(x, y, p).flatten()
            return csc_matrix((data, self._rowvals, self._colptrs))
        return sparse_cb
    else:
        return cb