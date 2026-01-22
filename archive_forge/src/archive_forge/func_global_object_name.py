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
def global_object_name(obj: Any) -> str:
    """
    Return full name of a global object.

    >>> from scrapy import Request
    >>> global_object_name(Request)
    'scrapy.http.request.Request'
    """
    return f'{obj.__module__}.{obj.__name__}'