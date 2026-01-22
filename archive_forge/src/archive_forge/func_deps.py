import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@property
def deps(self):
    """A tuple of the dependencies, other LazyArray instances, of this
        array.
        """
    return self._deps