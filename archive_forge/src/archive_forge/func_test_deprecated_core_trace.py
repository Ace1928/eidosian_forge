from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.trace import Tr
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_deprecated_core_trace():
    with warns_deprecated_sympy():
        from sympy.core.trace import Tr