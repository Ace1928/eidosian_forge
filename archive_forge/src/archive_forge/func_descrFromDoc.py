import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def descrFromDoc(obj):
    """
    Generate an appropriate description from docstring of the given object
    """
    if obj.__doc__ is None or obj.__doc__.isspace():
        return None
    lines = [x.strip() for x in obj.__doc__.split('\n') if x and (not x.isspace())]
    return ' '.join(lines)