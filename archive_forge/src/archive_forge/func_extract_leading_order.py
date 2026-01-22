from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from operator import attrgetter
from .basic import Basic
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .numbers import ilcm, igcd, equal_valued
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift
from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
@cacheit
def extract_leading_order(self, symbols, point=None):
    """
        Returns the leading term and its order.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (x + 1 + 1/x**5).extract_leading_order(x)
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order(x)
        ((1, O(1)),)
        >>> (x + x**2).extract_leading_order(x)
        ((x, O(x)),)

        """
    from sympy.series.order import Order
    lst = []
    symbols = list(symbols if is_sequence(symbols) else [symbols])
    if not point:
        point = [0] * len(symbols)
    seq = [(f, Order(f, *zip(symbols, point))) for f in self.args]
    for ef, of in seq:
        for e, o in lst:
            if o.contains(of) and o != of:
                of = None
                break
        if of is None:
            continue
        new_lst = [(ef, of)]
        for e, o in lst:
            if of.contains(o) and o != of:
                continue
            new_lst.append((e, o))
        lst = new_lst
    return tuple(lst)