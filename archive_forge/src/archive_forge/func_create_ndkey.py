from collections import defaultdict
from operator import itemgetter
from .dimension import Dimension
from .util import merge_dimensions
def create_ndkey(length, indexes, values):
    key = [None] * length
    for i, v in zip(indexes, values):
        key[i] = v
    return tuple(key)