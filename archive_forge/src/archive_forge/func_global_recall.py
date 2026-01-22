import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def global_recall(self):
    if self.global_true_positives + self.global_false_negatives > 0:
        return float(self.global_true_positives) / (self.global_true_positives + self.global_false_negatives)
    else:
        return 0.0