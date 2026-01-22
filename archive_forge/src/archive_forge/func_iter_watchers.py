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
def iter_watchers(self):
    """Iterator/generator over all the currently maintained watchers."""
    for _cb_metrics, watcher in self._watchers:
        yield watcher