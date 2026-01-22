import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def dynamic_programming(inputs, output, size_dict, memory_limit=None, **kwargs):
    optimizer = DynamicProgramming(**kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)