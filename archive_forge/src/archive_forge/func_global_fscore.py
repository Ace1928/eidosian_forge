import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def global_fscore(self):
    if self.global_precision + self.global_recall > 0:
        return 2 * self.global_precision * self.global_recall / (self.global_precision + self.global_recall)
    else:
        return 0.0