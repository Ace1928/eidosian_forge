import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
def parse_slice(useslice):
    """Parses the argument string "useslice" as a shard index and number and
    returns a function that filters on those arguments. i.e. input
    useslice="1:3" leads to output something like `lambda x: zlib.crc32(x) % 3
    == 1`.
    """
    if callable(useslice):
        return useslice
    if not useslice:
        return lambda x: True
    try:
        index, count = useslice.split(':')
        index = int(index)
        count = int(count)
    except Exception:
        msg = "Expected arguments shard index and count to follow option `-j i:t`, where i is the shard number and t is the total number of shards, found '%s'" % useslice
        raise ValueError(msg)
    if count == 0:
        return lambda x: True
    elif count < 0 or index < 0 or index >= count:
        raise ValueError('Sharding out of range')
    else:

        def decide(test):
            func = getattr(test, test._testMethodName)
            if 'always_test' in getattr(func, 'tags', {}):
                return True
            return abs(zlib.crc32(test.id().encode('utf-8'))) % count == index
        return decide