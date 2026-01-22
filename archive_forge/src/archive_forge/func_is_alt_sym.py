from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def is_alt_sym(self, eps=0.05, _random_prec=None):
    """Monte Carlo test for the symmetric/alternating group for degrees
        >= 8.

        Explanation
        ===========

        More specifically, it is one-sided Monte Carlo with the
        answer True (i.e., G is symmetric/alternating) guaranteed to be
        correct, and the answer False being incorrect with probability eps.

        For degree < 8, the order of the group is checked so the test
        is deterministic.

        Notes
        =====

        The algorithm itself uses some nontrivial results from group theory and
        number theory:
        1) If a transitive group ``G`` of degree ``n`` contains an element
        with a cycle of length ``n/2 < p < n-2`` for ``p`` a prime, ``G`` is the
        symmetric or alternating group ([1], pp. 81-82)
        2) The proportion of elements in the symmetric/alternating group having
        the property described in 1) is approximately `\\log(2)/\\log(n)`
        ([1], p.82; [2], pp. 226-227).
        The helper function ``_check_cycles_alt_sym`` is used to
        go over the cycles in a permutation and look for ones satisfying 1).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_alt_sym()
        False

        See Also
        ========

        _check_cycles_alt_sym

        """
    if _random_prec is not None:
        N_eps = _random_prec['N_eps']
        perms = (_random_prec[i] for i in range(N_eps))
        return self._eval_is_alt_sym_monte_carlo(perms=perms)
    if self._is_sym or self._is_alt:
        return True
    if self._is_sym is False and self._is_alt is False:
        return False
    n = self.degree
    if n < 8:
        return self._eval_is_alt_sym_naive()
    elif self.is_transitive():
        return self._eval_is_alt_sym_monte_carlo(eps=eps)
    self._is_sym, self._is_alt = (False, False)
    return False