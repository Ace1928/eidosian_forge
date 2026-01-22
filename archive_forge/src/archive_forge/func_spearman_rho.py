import math
from itertools import islice
from nltk.util import choose, ngrams
def spearman_rho(worder, normalize=True):
    """
    Calculates the Spearman's Rho correlation coefficient given the *worder*
    list of word alignment from word_rank_alignment(), using the formula:

        rho = 1 - sum(d**2) / choose(len(worder)+1, 3)

    Given that d is the sum of difference between the *worder* list of indices
    and the original word indices from the reference sentence.

    Using the (H0,R0) and (H5, R5) example from the paper

        >>> worder =  [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]
        >>> round(spearman_rho(worder, normalize=False), 3)
        -0.591
        >>> round(spearman_rho(worder), 3)
        0.205

    :param worder: The worder list output from word_rank_alignment
    :param type: list(int)
    """
    worder_len = len(worder)
    sum_d_square = sum(((wi - i) ** 2 for wi, i in zip(worder, range(worder_len))))
    rho = 1 - sum_d_square / choose(worder_len + 1, 3)
    if normalize:
        return (rho + 1) / 2
    else:
        return rho