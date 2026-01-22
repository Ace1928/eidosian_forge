import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def repr_key(k: Tuple[object, Callable, Tuple[Tuple[str, type], ...]]) -> str:
    owner_key, function, bindings = k
    return '%s.%s(injecting %s)' % (tuple(map(_describe, k[:2])) + (dict(k[2]),))