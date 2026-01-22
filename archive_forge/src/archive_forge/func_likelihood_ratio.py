import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@classmethod
def likelihood_ratio(cls, *marginals):
    """Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4."""
    cont = cls._contingency(*marginals)
    return 2 * sum((obs * _ln(obs / (exp + _SMALL) + _SMALL) for obs, exp in zip(cont, cls._expected_values(cont))))