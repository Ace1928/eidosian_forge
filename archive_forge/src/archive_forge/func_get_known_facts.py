from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
from sympy.logic.inference import satisfiable
@cacheit
def get_known_facts(x=None):
    """
    Facts between unary predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')
    fact = And(Exclusive(Q.negative_infinite(x), Q.negative(x), Q.zero(x), Q.positive(x), Q.positive_infinite(x)), Exclusive(Q.real(x), Q.imaginary(x)), Implies(Q.real(x) | Q.imaginary(x), Q.complex(x)), Exclusive(Q.transcendental(x), Q.algebraic(x)), Equivalent(Q.real(x), Q.rational(x) | Q.irrational(x)), Exclusive(Q.irrational(x), Q.rational(x)), Implies(Q.rational(x), Q.algebraic(x)), Exclusive(Q.even(x), Q.odd(x)), Implies(Q.integer(x), Q.rational(x)), Implies(Q.zero(x), Q.even(x)), Exclusive(Q.composite(x), Q.prime(x)), Implies(Q.composite(x) | Q.prime(x), Q.integer(x) & Q.positive(x)), Implies(Q.even(x) & Q.positive(x) & ~Q.prime(x), Q.composite(x)), Implies(Q.real(x), Q.hermitian(x)), Implies(Q.imaginary(x), Q.antihermitian(x)), Implies(Q.zero(x), Q.hermitian(x) | Q.antihermitian(x)), Exclusive(Q.infinite(x), Q.finite(x)), Implies(Q.complex(x), Q.finite(x)), Implies(Q.negative_infinite(x) | Q.positive_infinite(x), Q.infinite(x)), Implies(Q.finite(x) | Q.infinite(x), Q.commutative(x)), Implies(Q.orthogonal(x), Q.positive_definite(x)), Implies(Q.orthogonal(x), Q.unitary(x)), Implies(Q.unitary(x) & Q.real_elements(x), Q.orthogonal(x)), Implies(Q.unitary(x), Q.normal(x)), Implies(Q.unitary(x), Q.invertible(x)), Implies(Q.normal(x), Q.square(x)), Implies(Q.diagonal(x), Q.normal(x)), Implies(Q.positive_definite(x), Q.invertible(x)), Implies(Q.diagonal(x), Q.upper_triangular(x)), Implies(Q.diagonal(x), Q.lower_triangular(x)), Implies(Q.lower_triangular(x), Q.triangular(x)), Implies(Q.upper_triangular(x), Q.triangular(x)), Implies(Q.triangular(x), Q.upper_triangular(x) | Q.lower_triangular(x)), Implies(Q.upper_triangular(x) & Q.lower_triangular(x), Q.diagonal(x)), Implies(Q.diagonal(x), Q.symmetric(x)), Implies(Q.unit_triangular(x), Q.triangular(x)), Implies(Q.invertible(x), Q.fullrank(x)), Implies(Q.invertible(x), Q.square(x)), Implies(Q.symmetric(x), Q.square(x)), Implies(Q.fullrank(x) & Q.square(x), Q.invertible(x)), Equivalent(Q.invertible(x), ~Q.singular(x)), Implies(Q.integer_elements(x), Q.real_elements(x)), Implies(Q.real_elements(x), Q.complex_elements(x)))
    return fact