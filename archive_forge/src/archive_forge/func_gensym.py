import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
def gensym():
    """
    Create a unique Python identifier.
    """
    return '_symbol_' + str(next(counter))