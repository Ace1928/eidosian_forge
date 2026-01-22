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
def find_instantiations(self, func):
    """
        Find substitutions of the free symbols that match ``func``.

        Return the substitution dictionaries as a list. Note that the returned
        instantiations need not actually match, or be valid!

        """
    from sympy.solvers import solve
    ap = func.ap
    bq = func.bq
    if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
        raise TypeError('Cannot instantiate other number of parameters')
    symbol_values = []
    for a in self.symbols:
        if a in self.func.ap.args:
            symbol_values.append(ap)
        elif a in self.func.bq.args:
            symbol_values.append(bq)
        else:
            raise ValueError('At least one of the parameters of the formula must be equal to %s' % (a,))
    base_repl = [dict(list(zip(self.symbols, values))) for values in product(*symbol_values)]
    abuckets, bbuckets = [sift(params, _mod1) for params in [ap, bq]]
    a_inv, b_inv = [{a: len(vals) for a, vals in bucket.items()} for bucket in [abuckets, bbuckets]]
    critical_values = [[0] for _ in self.symbols]
    result = []
    _n = Dummy()
    for repl in base_repl:
        symb_a, symb_b = [sift(params, lambda x: _mod1(x.xreplace(repl))) for params in [self.func.ap, self.func.bq]]
        for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if mod not in bucket or mod not in obucket or len(bucket[mod]) != len(obucket[mod]):
                    break
                for a, vals in zip(self.symbols, critical_values):
                    if repl[a].free_symbols:
                        continue
                    exprs = [expr for expr in obucket[mod] if expr.has(a)]
                    repl0 = repl.copy()
                    repl0[a] += _n
                    for expr in exprs:
                        for target in bucket[mod]:
                            n0, = solve(expr.xreplace(repl0) - target, _n)
                            if n0.free_symbols:
                                raise ValueError('Value should not be true')
                            vals.append(n0)
        else:
            values = []
            for a, vals in zip(self.symbols, critical_values):
                a0 = repl[a]
                min_ = floor(min(vals))
                max_ = ceiling(max(vals))
                values.append([a0 + n for n in range(min_, max_ + 1)])
            result.extend((dict(list(zip(self.symbols, l))) for l in product(*values)))
    return result