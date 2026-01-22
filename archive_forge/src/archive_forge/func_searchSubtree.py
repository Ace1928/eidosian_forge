import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def searchSubtree(self, begin):
    """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
    end = begin + 1
    total = self[begin].arity
    while total > 0:
        total += self[end].arity - 1
        end += 1
    return slice(begin, end)