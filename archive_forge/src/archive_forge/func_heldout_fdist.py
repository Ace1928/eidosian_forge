import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def heldout_fdist(self):
    """
        Return the heldout frequency distribution that this
        probability distribution is based on.

        :rtype: FreqDist
        """
    return self._heldout_fdist