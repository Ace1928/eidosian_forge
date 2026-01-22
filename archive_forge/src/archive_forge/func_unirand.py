import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
@classmethod
def unirand(cls, samples):
    """
        The key function that creates a randomized initial distribution
        that still sums to 1. Set as a dictionary of prob values so that
        it can still be passed to MutableProbDist and called with identical
        syntax to UniformProbDist
        """
    samples = set(samples)
    randrow = [random.random() for i in range(len(samples))]
    total = sum(randrow)
    for i, x in enumerate(randrow):
        randrow[i] = x / total
    total = sum(randrow)
    if total != 1:
        randrow[-1] -= total - 1
    return {s: randrow[i] for i, s in enumerate(samples)}