import abc
from typing import Tuple
import numpy as np
import cvxpy.interface.matrix_utilities
@staticmethod
def scalar_const(converter):

    def new_converter(self, value, convert_scalars: bool=False):
        if not convert_scalars and cvxpy.interface.matrix_utilities.is_scalar(value):
            return cvxpy.interface.matrix_utilities.scalar_value(value)
        else:
            return converter(self, value)
    return new_converter