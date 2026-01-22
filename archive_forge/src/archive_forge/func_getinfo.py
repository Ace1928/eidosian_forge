import collections
import collections.abc
import functools
import heapq
import random
import time
from . import keys
def getinfo():
    nonlocal hits, misses
    return _CacheInfo(hits, misses, 0, 0)