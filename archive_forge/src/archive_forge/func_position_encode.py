import numpy
from .. import registry
from ..compat import cublas, cupy, cupyx
from ..types import DeviceTypes
from ..util import (
from . import _custom_kernels
from .numpy_ops import NumpyOps
from .ops import Ops
def position_encode(self, N, D, period=10000, out=None):
    positions = NumpyOps().position_encode(N, D, period=period, out=out)
    return self.asarray(positions)