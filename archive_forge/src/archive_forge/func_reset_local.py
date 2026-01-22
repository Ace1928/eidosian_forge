import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def reset_local(self):
    """Resets the local portion of the internal evaluation results to initial state."""
    self.num_inst = 0.0
    self.lcm = numpy.zeros((self.k, self.k))