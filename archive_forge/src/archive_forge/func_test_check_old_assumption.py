from sympy.abc import t, w, x, y, z, n, k, m, p, i
from sympy.assumptions import (ask, AssumptionsContext, Q, register_handler,
from sympy.assumptions.assume import assuming, global_assumptions, Predicate
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.facts import (single_fact_lookup,
from sympy.assumptions.handlers import AskHandler
from sympy.assumptions.ask_generated import (get_all_known_facts,
from sympy.core.add import Add
from sympy.core.numbers import (I, Integer, Rational, oo, zoo, pi)
from sympy.core.singleton import S
from sympy.core.power import Pow
from sympy.core.symbol import Str, symbols, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.logic.boolalg import Equivalent, Implies, Xor, And, to_cnf
from sympy.matrices import Matrix, SparseMatrix
from sympy.testing.pytest import (XFAIL, slow, raises, warns_deprecated_sympy,
import math
def test_check_old_assumption():
    x = symbols('x', real=True)
    assert ask(Q.real(x)) is True
    assert ask(Q.imaginary(x)) is False
    assert ask(Q.complex(x)) is True
    x = symbols('x', imaginary=True)
    assert ask(Q.real(x)) is False
    assert ask(Q.imaginary(x)) is True
    assert ask(Q.complex(x)) is True
    x = symbols('x', complex=True)
    assert ask(Q.real(x)) is None
    assert ask(Q.complex(x)) is True
    x = symbols('x', positive=True)
    assert ask(Q.positive(x)) is True
    assert ask(Q.negative(x)) is False
    assert ask(Q.real(x)) is True
    x = symbols('x', commutative=False)
    assert ask(Q.commutative(x)) is False
    x = symbols('x', negative=True)
    assert ask(Q.positive(x)) is False
    assert ask(Q.negative(x)) is True
    x = symbols('x', nonnegative=True)
    assert ask(Q.negative(x)) is False
    assert ask(Q.positive(x)) is None
    assert ask(Q.zero(x)) is None
    x = symbols('x', finite=True)
    assert ask(Q.finite(x)) is True
    x = symbols('x', prime=True)
    assert ask(Q.prime(x)) is True
    assert ask(Q.composite(x)) is False
    x = symbols('x', composite=True)
    assert ask(Q.prime(x)) is False
    assert ask(Q.composite(x)) is True
    x = symbols('x', even=True)
    assert ask(Q.even(x)) is True
    assert ask(Q.odd(x)) is False
    x = symbols('x', odd=True)
    assert ask(Q.even(x)) is False
    assert ask(Q.odd(x)) is True
    x = symbols('x', nonzero=True)
    assert ask(Q.nonzero(x)) is True
    assert ask(Q.zero(x)) is False
    x = symbols('x', zero=True)
    assert ask(Q.zero(x)) is True
    x = symbols('x', integer=True)
    assert ask(Q.integer(x)) is True
    x = symbols('x', rational=True)
    assert ask(Q.rational(x)) is True
    assert ask(Q.irrational(x)) is False
    x = symbols('x', irrational=True)
    assert ask(Q.irrational(x)) is True
    assert ask(Q.rational(x)) is False