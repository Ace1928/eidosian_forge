from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
@staticmethod
def format_references(func):
    """ Creates a string representation of referenced callbacks.
        Returns:
            str that represents a callback reference.
        """
    try:
        return func.__name__
    except AttributeError:
        pass
    if isinstance(func, partial):
        return '%s(%s)' % (func.func.__name__, ', '.join(itertools.chain((str(_) for _ in func.args), ('%s=%s' % (key, value) for key, value in iteritems(func.keywords if func.keywords else {})))))
    return str(func)