from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrices import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module
def numpy_to_sympy(m, **options):
    """Convert a numpy matrix to a SymPy matrix."""
    return MatrixBase(m)