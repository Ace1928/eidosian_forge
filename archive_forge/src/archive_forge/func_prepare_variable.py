from __future__ import annotations
import copy
import numpy as np
from xarray.backends.common import AbstractWritableDataStore
from xarray.core.variable import Variable
def prepare_variable(self, k, v, *args, **kwargs):
    new_var = Variable(v.dims, np.empty_like(v), v.attrs)
    self._variables[k] = new_var
    return (new_var, v.data)