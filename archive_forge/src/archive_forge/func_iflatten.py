import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def iflatten(x: Iterable) -> Iterable:
    """iflatten(sequence) -> iterator

    Similar to ``.flatten()``, but returns iterator instead"""
    for el in x:
        if is_listlike(el):
            for el_ in iflatten(el):
                yield el_
        else:
            yield el