from sympy.core import symbols, Lambda
from sympy.functions import KroneckerDelta
from sympy.matrices import Matrix
from sympy.matrices.expressions import FunctionMatrix, MatrixExpr, Identity
from sympy.testing.pytest import raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
def test_replace_issue():
    X = FunctionMatrix(3, 3, KroneckerDelta)
    assert X.replace(lambda x: True, lambda x: x) == X