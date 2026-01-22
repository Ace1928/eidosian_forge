import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from itertools import accumulate
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary
@abstractmethod
def unmasked_score(self, word, context=None):
    """Score a word given some optional context.

        Concrete models are expected to provide an implementation.
        Note that this method does not mask its arguments with the OOV label.
        Use the `score` method for that.

        :param str word: Word for which we want the score
        :param tuple(str) context: Context the word is in.
            If `None`, compute unigram score.
        :param context: tuple(str) or None
        :rtype: float
        """
    raise NotImplementedError()