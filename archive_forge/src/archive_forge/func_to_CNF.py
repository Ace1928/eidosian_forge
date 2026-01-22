from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)
@classmethod
def to_CNF(cls, expr):
    from sympy.assumptions.facts import get_composite_predicates
    expr = to_NNF(expr, get_composite_predicates())
    expr = distribute_AND_over_OR(expr)
    return expr