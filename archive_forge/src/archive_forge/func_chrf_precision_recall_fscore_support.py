import re
from collections import Counter, defaultdict
from nltk.util import ngrams
def chrf_precision_recall_fscore_support(reference, hypothesis, n, beta=3.0, epsilon=1e-16):
    """
    This function computes the precision, recall and fscore from the ngram
    overlaps. It returns the `support` which is the true positive score.

    By underspecifying the input type, the function will be agnostic as to how
    it computes the ngrams and simply take the whichever element in the list;
    it could be either token or character.

    :param reference: The reference sentence.
    :type reference: list
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: list
    :param n: Extract up to the n-th order ngrams
    :type n: int
    :param beta: The parameter to assign more importance to recall over precision.
    :type beta: float
    :param epsilon: The fallback value if the hypothesis or reference is empty.
    :type epsilon: float
    :return: Returns the precision, recall and f-score and support (true positive).
    :rtype: tuple(float)
    """
    ref_ngrams = Counter(ngrams(reference, n))
    hyp_ngrams = Counter(ngrams(hypothesis, n))
    overlap_ngrams = ref_ngrams & hyp_ngrams
    tp = sum(overlap_ngrams.values())
    tpfp = sum(hyp_ngrams.values())
    tpfn = sum(ref_ngrams.values())
    try:
        prec = tp / tpfp
        rec = tp / tpfn
        factor = beta ** 2
        fscore = (1 + factor) * (prec * rec) / (factor * prec + rec)
    except ZeroDivisionError:
        prec = rec = fscore = epsilon
    return (prec, rec, fscore, tp)