import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def feval(label, pred):
    """Internal eval function."""
    return numpy_feval(label, pred)