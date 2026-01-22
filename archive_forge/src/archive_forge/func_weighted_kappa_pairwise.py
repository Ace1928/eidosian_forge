import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def weighted_kappa_pairwise(self, cA, cB, max_distance=1.0):
    """Cohen 1968"""
    total = 0.0
    label_freqs = ConditionalFreqDist(((x['coder'], x['labels']) for x in self.data if x['coder'] in (cA, cB)))
    for j in self.K:
        for l in self.K:
            total += label_freqs[cA][j] * label_freqs[cB][l] * self.distance(j, l)
    De = total / (max_distance * pow(len(self.I), 2))
    log.debug('Expected disagreement between %s and %s: %f', cA, cB, De)
    Do = self.Do_Kw_pairwise(cA, cB)
    ret = 1.0 - Do / De
    return ret