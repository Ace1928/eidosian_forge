import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def get_t(self, x):
    s = -1 if x < 0 else 1
    return s / (abs(x) + 1)