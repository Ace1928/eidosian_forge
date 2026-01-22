from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.gate import (
def get_target_matrix(self, format='sympy'):
    if format == 'sympy':
        return Matrix([[1, 0], [0, exp(Integer(2) * pi * I / Integer(2) ** self.k)]])
    raise NotImplementedError('Invalid format for the R_k gate: %r' % format)