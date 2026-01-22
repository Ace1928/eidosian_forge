from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational
def test_syzygy():
    R = QQ.old_poly_ring(x, y, z)
    M = R.free_module(1).submodule([x * y], [y * z], [x * z])
    S = R.free_module(3).submodule([0, x, -y], [z, -x, 0])
    assert M.syzygy_module() == S
    M2 = M / ([x * y * z],)
    S2 = R.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M2.syzygy_module() == S2
    F = R.free_module(3)
    assert F.submodule(*F.basis()).syzygy_module() == F.submodule()
    R2 = QQ.old_poly_ring(x, y, z) / [x * y * z]
    M3 = R2.free_module(1).submodule([x * y], [y * z], [x * z])
    S3 = R2.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    assert M3.syzygy_module() == S3