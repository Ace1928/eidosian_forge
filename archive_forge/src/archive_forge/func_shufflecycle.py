from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
def shufflecycle(it):
    it = list(it)
    shuffle = random.shuffle
    for _ in repeat(None):
        shuffle(it)
        yield it[0]