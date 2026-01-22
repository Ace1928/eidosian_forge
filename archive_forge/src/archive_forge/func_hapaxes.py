import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def hapaxes(self):
    """
        Return a list of all samples that occur once (hapax legomena)

        :rtype: list
        """
    return [item for item in self if self[item] == 1]