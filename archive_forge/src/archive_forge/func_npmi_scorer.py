import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Not used.
    min_count: int
        Ignore all bigrams with total collected count lower than this value.
    corpus_word_count : int
        Total number of words in the corpus.

    Returns
    -------
    float
        If bigram_count >= min_count, return the collocation score, in the range -1 to 1.
        Otherwise return -inf.

    Notes
    -----
    Formula: :math:`\\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \\frac{word\\_count}{corpus\\_word\\_count}`

    """
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        try:
            return log(pab / (pa * pb)) / -log(pab)
        except ValueError:
            return NEGATIVE_INFINITY
    else:
        return NEGATIVE_INFINITY