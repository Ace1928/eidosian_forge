import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def nbest(self, score_fn, n):
    """Returns the top n ngrams when scored by the given function."""
    return [p for p, s in self.score_ngrams(score_fn)[:n]]