import sys
from collections import defaultdict
from nltk.classify.util import accuracy as eval_accuracy
from nltk.classify.util import apply_features
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import f_measure as eval_f_measure
from nltk.metrics import precision as eval_precision
from nltk.metrics import recall as eval_recall
from nltk.probability import FreqDist
def unigram_word_feats(self, words, top_n=None, min_freq=0):
    """
        Return most common top_n word features.

        :param words: a list of words/tokens.
        :param top_n: number of best words/tokens to use, sorted by frequency.
        :rtype: list(str)
        :return: A list of `top_n` words/tokens (with no duplicates) sorted by
            frequency.
        """
    unigram_feats_freqs = FreqDist((word for word in words))
    return [w for w, f in unigram_feats_freqs.most_common(top_n) if unigram_feats_freqs[w] > min_freq]