import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from itertools import accumulate
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary
def logscore(self, word, context=None):
    """Evaluate the log score of this word in this context.

        The arguments are the same as for `score` and `unmasked_score`.

        """
    return log_base2(self.score(word, context))