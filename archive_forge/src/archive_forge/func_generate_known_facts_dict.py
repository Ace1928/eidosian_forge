from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
from sympy.logic.inference import satisfiable
def generate_known_facts_dict(keys, fact):
    """
    Computes and returns a dictionary which contains the relations between
    unary predicates.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    All predicates in *keys* and *fact* must be unary and have same placeholder
    symbol.

    Parameters
    ==========

    keys : list of AppliedPredicate instances.

    fact : Fact between predicates in conjugated normal form.

    Examples
    ========

    >>> from sympy import Q, And, Implies
    >>> from sympy.assumptions.facts import generate_known_facts_dict
    >>> from sympy.abc import x
    >>> keys = [Q.even(x), Q.odd(x), Q.zero(x)]
    >>> fact = And(Implies(Q.even(x), ~Q.odd(x)),
    ...     Implies(Q.zero(x), Q.even(x)))
    >>> generate_known_facts_dict(keys, fact)
    {Q.even: ({Q.even}, {Q.odd}),
     Q.odd: ({Q.odd}, {Q.even, Q.zero}),
     Q.zero: ({Q.even, Q.zero}, {Q.odd})}
    """
    fact_cnf = to_cnf(fact)
    mapping = single_fact_lookup(keys, fact_cnf)
    ret = {}
    for key, value in mapping.items():
        implied = set()
        rejected = set()
        for expr in value:
            if isinstance(expr, AppliedPredicate):
                implied.add(expr.function)
            elif isinstance(expr, Not):
                pred = expr.args[0]
                rejected.add(pred.function)
        ret[key.function] = (implied, rejected)
    return ret