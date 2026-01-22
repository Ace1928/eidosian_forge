import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from itertools import accumulate
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary
def context_counts(self, context):
    """Helper method for retrieving counts for a given context.

        Assumes context has been checked and oov words in it masked.
        :type context: tuple(str) or None

        """
    return self.counts[len(context) + 1][context] if context else self.counts.unigrams