import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_fn_frequencies(self):
    """Get a dictionary mapping function names to the number of times they
        are used in the computational graph.
        """
    return self.history_stats('count')