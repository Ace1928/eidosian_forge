import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_max_size(self):
    """Get the largest single tensor size appearing in this computation."""
    return max((node.size for node in self.descend()))