import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def idata(iterable, index_len=None):
    """
    Method decorator to add to your test methods.

    Should be added to methods of instances of ``unittest.TestCase``.

    :param iterable: iterable of the values to provide to the test function.
    :param index_len: an optional integer specifying the width to zero-pad the
        test identifier indices to.  If not provided, this will add the fewest
        zeros necessary to make all identifiers the same length.
    """
    if index_len is None:
        iterable = tuple(iterable)
        index_len = len(str(len(iterable)))

    def wrapper(func):
        setattr(func, DATA_ATTR, iterable)
        setattr(func, INDEX_LEN, index_len)
        return func
    return wrapper