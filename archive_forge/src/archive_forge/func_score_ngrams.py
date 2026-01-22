import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def score_ngrams(self, score_fn):
    """Returns a sequence of (ngram, score) pairs ordered from highest to
        lowest score, as determined by the scoring function provided.
        """
    return sorted(self._score_ngrams(score_fn), key=lambda t: (-t[1], t[0]))