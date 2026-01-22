from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def find_pure_symbol(symbols, unknown_clauses):
    """
    Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import find_pure_symbol
    >>> find_pure_symbol([A, B, D], [A|~B,~B|~D,D|A])
    (A, True)

    """
    for sym in symbols:
        found_pos, found_neg = (False, False)
        for c in unknown_clauses:
            if not found_pos and sym in disjuncts(c):
                found_pos = True
            if not found_neg and Not(sym) in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return (sym, found_pos)
    return (None, None)