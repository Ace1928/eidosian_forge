from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def best_path_simple(self, unlabeled_sequence):
    """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.  This uses a simple, direct method, and is included for
        teaching purposes.

        :return: the state sequence
        :rtype: sequence of any
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
    unlabeled_sequence = self._transform(unlabeled_sequence)
    return self._best_path_simple(unlabeled_sequence)