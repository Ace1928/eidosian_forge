import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def reset_micro(self):
    self._sse_p = 0
    self._mean_p = 0
    self._sse_l = 0
    self._mean_l = 0
    self._pred_nums = 0
    self._label_nums = 0
    self._conv = 0