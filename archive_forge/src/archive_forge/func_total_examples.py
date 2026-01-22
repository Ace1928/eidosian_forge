import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
@property
def total_examples(self):
    return self.false_negatives + self.false_positives + self.true_negatives + self.true_positives