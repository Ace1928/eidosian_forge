from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def point_entropy(self, unlabeled_sequence):
    """
        Returns the pointwise entropy over the possible states at each
        position in the chain, given the observation sequence.
        """
    unlabeled_sequence = self._transform(unlabeled_sequence)
    T = len(unlabeled_sequence)
    N = len(self._states)
    alpha = self._forward_probability(unlabeled_sequence)
    beta = self._backward_probability(unlabeled_sequence)
    normalisation = logsumexp2(alpha[T - 1])
    entropies = np.zeros(T, np.float64)
    probs = np.zeros(N, np.float64)
    for t in range(T):
        for s in range(N):
            probs[s] = alpha[t, s] + beta[t, s] - normalisation
        for s in range(N):
            entropies[t] -= 2 ** probs[s] * probs[s]
    return entropies