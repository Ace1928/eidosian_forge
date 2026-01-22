import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def softmax_sequences(self, Xs: Floats2d, lengths: Ints1d, *, inplace: bool=False, axis: int=-1) -> Floats2d:
    if Xs.ndim >= 3:
        err = f'Softmax currently only supports 2d. Got: {Xs.ndim}'
        raise NotImplementedError(err)
    Xs = self.xp.clip(Xs, -20.0, 20.0)
    new_x = self.xp.exp(Xs)
    summed = self.backprop_reduce_sum(self.reduce_sum(new_x, lengths), lengths)
    new_x /= summed
    return new_x