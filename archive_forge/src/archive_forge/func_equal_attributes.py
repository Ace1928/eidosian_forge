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
def equal_attributes(obj1: Any, obj2: Any, attributes: Optional[List[Union[str, Callable]]]) -> bool:
    """Compare two objects attributes"""
    if not attributes:
        return False
    temp1, temp2 = (object(), object())
    for attr in attributes:
        if callable(attr):
            if attr(obj1) != attr(obj2):
                return False
        elif getattr(obj1, attr, temp1) != getattr(obj2, attr, temp2):
            return False
    return True