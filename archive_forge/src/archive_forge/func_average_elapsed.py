import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
@property
def average_elapsed(self):
    """Avg. amount of time the periodic callback has ran for.

        This may raise a ``ZeroDivisionError`` if there has been no runs.
        """
    return self._metrics['elapsed'] / self._metrics['runs']