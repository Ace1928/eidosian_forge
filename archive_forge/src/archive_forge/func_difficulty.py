from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def difficulty(self, func):
    """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
    if self.gamma != func.gamma:
        return -1
    oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for params in (self.ap, self.bq, func.ap, func.bq)]
    diff = 0
    for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
        for mod in set(list(bucket.keys()) + list(obucket.keys())):
            if mod not in bucket or mod not in obucket or len(bucket[mod]) != len(obucket[mod]):
                return -1
            l1 = list(bucket[mod])
            l2 = list(obucket[mod])
            l1.sort()
            l2.sort()
            for i, j in zip(l1, l2):
                diff += abs(i - j)
    return diff