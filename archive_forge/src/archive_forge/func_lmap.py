import types
import sys
import numbers
import functools
import copy
import inspect
def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))