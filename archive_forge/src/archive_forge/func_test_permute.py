from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.trace import Tr
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_permute():
    A, B, C, D, E, F, G = symbols('A B C D E F G', commutative=False)
    t = Tr(A * B * C * D * E * F * G)
    assert t.permute(0).args[0].args == (A, B, C, D, E, F, G)
    assert t.permute(2).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(4).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(6).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(8).args[0].args == t.permute(1).args[0].args
    assert t.permute(-1).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(-3).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(-5).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(-8).args[0].args == t.permute(-1).args[0].args
    t = Tr((A + B) * (B * B) * C * D)
    assert t.permute(2).args[0].args == (C, D, A + B, B ** 2)
    t1 = Tr(A * B)
    t2 = t1.permute(1)
    assert id(t1) != id(t2) and t1 == t2