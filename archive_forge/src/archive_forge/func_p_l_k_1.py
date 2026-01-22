from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
def p_l_k_1(self):
    """Relative frequency of each L value over all possible single rows"""
    ranks = range(1, self.k + 1)
    rank_perms = np.array(list(permutations(ranks)))
    Ls = (ranks * rank_perms).sum(axis=1)
    counts = np.histogram(Ls, np.arange(self.a - 0.5, self.b + 1.5))[0]
    return counts / math.factorial(self.k)