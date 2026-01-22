import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def set_logprob(self, prob):
    raise ValueError('%s is immutable' % self.__class__.__name__)