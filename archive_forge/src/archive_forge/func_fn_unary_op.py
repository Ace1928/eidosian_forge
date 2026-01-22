import numbers
from functools import reduce
from operator import mul
import numpy as np
def fn_unary_op(self):
    try:
        return self._op(op)
    except SystemError as e:
        message = "Numpy returned an uninformative error. It possibly should be 'Integers to negative integer powers are not allowed.' See https://github.com/numpy/numpy/issues/19634 for details."
        raise ValueError(message) from e