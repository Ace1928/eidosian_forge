from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__pow__():
    assert QQ(1) ** 10 == QQ(1)
    assert QQ(2) ** 10 == QQ(1024)
    assert QQ(1) ** (-10) == QQ(1)
    assert QQ(2) ** (-10) == QQ(1, 1024)