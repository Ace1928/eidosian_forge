import numpy
from .. import registry
from ..compat import cublas, cupy, cupyx
from ..types import DeviceTypes
from ..util import (
from . import _custom_kernels
from .numpy_ops import NumpyOps
from .ops import Ops
def backprop_relu(self, dY, Y, inplace=False):
    if not inplace:
        return dY * (Y > 0)
    dY *= Y > 0
    return dY