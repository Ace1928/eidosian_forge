from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int
def eliminate_word(self, gen, by=None, _all=False, inverse=True):
    """
        For an associative word `self`, a subword `gen`, and an associative
        word `by` (identity by default), return the associative word obtained by
        replacing each occurrence of `gen` in `self` by `by`. If `_all = True`,
        the occurrences of `gen` that may appear after the first substitution will
        also be replaced and so on until no occurrences are found. This might not
        always terminate (e.g. `(x).eliminate_word(x, x**2, _all=True)`).

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> w = x**5*y*x**2*y**-4*x
        >>> w.eliminate_word( x, x**2 )
        x**10*y*x**4*y**-4*x**2
        >>> w.eliminate_word( x, y**-1 )
        y**-11
        >>> w.eliminate_word(x**5)
        y*x**2*y**-4*x
        >>> w.eliminate_word(x*y, y)
        x**4*y*x**2*y**-4*x

        See Also
        ========
        substituted_word

        """
    if by is None:
        by = self.group.identity
    if self.is_independent(gen) or gen == by:
        return self
    if gen == self:
        return by
    if gen ** (-1) == by:
        _all = False
    word = self
    l = len(gen)
    try:
        i = word.subword_index(gen)
        k = 1
    except ValueError:
        if not inverse:
            return word
        try:
            i = word.subword_index(gen ** (-1))
            k = -1
        except ValueError:
            return word
    word = word.subword(0, i) * by ** k * word.subword(i + l, len(word)).eliminate_word(gen, by)
    if _all:
        return word.eliminate_word(gen, by, _all=True, inverse=inverse)
    else:
        return word