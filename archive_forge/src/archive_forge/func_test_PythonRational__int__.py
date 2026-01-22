from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__int__():
    assert int(QQ(-1, 4)) == 0
    assert int(QQ(1, 4)) == 0
    assert int(QQ(-5, 4)) == -1
    assert int(QQ(5, 4)) == 1