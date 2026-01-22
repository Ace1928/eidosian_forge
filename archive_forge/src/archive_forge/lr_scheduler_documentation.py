import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0.