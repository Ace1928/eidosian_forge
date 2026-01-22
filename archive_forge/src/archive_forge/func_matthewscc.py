import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def matthewscc(self, use_global=False):
    """Calculate the Matthew's Correlation Coefficent"""
    if use_global:
        if not self.global_total_examples:
            return 0.0
        true_pos = float(self.global_true_positives)
        false_pos = float(self.global_false_positives)
        false_neg = float(self.global_false_negatives)
        true_neg = float(self.global_true_negatives)
    else:
        if not self.total_examples:
            return 0.0
        true_pos = float(self.true_positives)
        false_pos = float(self.false_positives)
        false_neg = float(self.false_negatives)
        true_neg = float(self.true_negatives)
    terms = [true_pos + false_pos, true_pos + false_neg, true_neg + false_pos, true_neg + false_neg]
    denom = 1.0
    for t in filter(lambda t: t != 0.0, terms):
        denom *= t
    return (true_pos * true_neg - false_pos * false_neg) / math.sqrt(denom)