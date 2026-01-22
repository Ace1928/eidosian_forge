import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def fisher(cls, *marginals):
    """Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less
        sensitive to small counts than PMI or Chi Sq, but also more expensive
        to compute. Requires scipy.
        """
    n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)
    odds, pvalue = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative='less')
    return pvalue