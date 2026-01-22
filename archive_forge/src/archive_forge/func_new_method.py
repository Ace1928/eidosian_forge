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
@wraps(method)
def new_method(self: Any, *args: Any, **kwargs: Any) -> Any:
    if self not in cache:
        cache[self] = method(self, *args, **kwargs)
    return cache[self]