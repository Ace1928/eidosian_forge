import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@staticmethod
def mi_like(*marginals, **kwargs):
    """Scores ngrams using a variant of mutual information. The keyword
        argument power sets an exponent (default 3) for the numerator. No
        logarithm of the result is calculated.
        """
    return marginals[NGRAM] ** kwargs.get('power', 3) / _product(marginals[UNIGRAMS])