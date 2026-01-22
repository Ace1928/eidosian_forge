import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def cost_memory_removed_jitter(size12, size1, size2, k12, k1, k2):
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)