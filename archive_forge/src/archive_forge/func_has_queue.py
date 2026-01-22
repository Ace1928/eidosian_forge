import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@property
def has_queue(self):
    """ Return boolean indicating if machine has queue or not """
    return self._queued