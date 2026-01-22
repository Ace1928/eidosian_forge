import numbers
from functools import reduce
from operator import mul
import numpy as np
@property
def is_sliced_view(self):
    return self._lengths.sum() != self._data.shape[0]