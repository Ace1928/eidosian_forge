import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def score_ngram(self, score_fn, w1, w2, w3, w4):
    n_all = self.N
    n_iiii = self.ngram_fd[w1, w2, w3, w4]
    if not n_iiii:
        return
    n_iiix = self.iii[w1, w2, w3]
    n_xiii = self.iii[w2, w3, w4]
    n_iixi = self.iixi[w1, w2, w4]
    n_ixii = self.ixii[w1, w3, w4]
    n_iixx = self.ii[w1, w2]
    n_xxii = self.ii[w3, w4]
    n_xiix = self.ii[w2, w3]
    n_ixix = self.ixi[w1, w3]
    n_ixxi = self.ixxi[w1, w4]
    n_xixi = self.ixi[w2, w4]
    n_ixxx = self.word_fd[w1]
    n_xixx = self.word_fd[w2]
    n_xxix = self.word_fd[w3]
    n_xxxi = self.word_fd[w4]
    return score_fn(n_iiii, (n_iiix, n_iixi, n_ixii, n_xiii), (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix), (n_ixxx, n_xixx, n_xxix, n_xxxi), n_all)