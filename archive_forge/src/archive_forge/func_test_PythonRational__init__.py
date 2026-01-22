from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__init__():
    assert QQ(0).numerator == 0
    assert QQ(0).denominator == 1
    assert QQ(0, 1).numerator == 0
    assert QQ(0, 1).denominator == 1
    assert QQ(0, -1).numerator == 0
    assert QQ(0, -1).denominator == 1
    assert QQ(1).numerator == 1
    assert QQ(1).denominator == 1
    assert QQ(1, 1).numerator == 1
    assert QQ(1, 1).denominator == 1
    assert QQ(-1, -1).numerator == 1
    assert QQ(-1, -1).denominator == 1
    assert QQ(-1).numerator == -1
    assert QQ(-1).denominator == 1
    assert QQ(-1, 1).numerator == -1
    assert QQ(-1, 1).denominator == 1
    assert QQ(1, -1).numerator == -1
    assert QQ(1, -1).denominator == 1
    assert QQ(1, 2).numerator == 1
    assert QQ(1, 2).denominator == 2
    assert QQ(3, 4).numerator == 3
    assert QQ(3, 4).denominator == 4
    assert QQ(2, 2).numerator == 1
    assert QQ(2, 2).denominator == 1
    assert QQ(2, 4).numerator == 1
    assert QQ(2, 4).denominator == 2