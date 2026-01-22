from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.dagger import adjoint, Dagger
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.operator import Operator, IdentityOperator
def test_eval_adjoint():
    f = Foo()
    d = Dagger(f)
    assert d == I