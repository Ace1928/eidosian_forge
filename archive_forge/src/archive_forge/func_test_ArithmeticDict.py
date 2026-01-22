from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict():
    d1 = ArithmeticDict(float, [('a', 1.0), ('b', 2.0)])
    d2 = ArithmeticDict(float, [('c', 5.0), ('b', 3.0)])
    d3 = d1 + d2
    assert d3['a'] == 1.0
    assert d3['b'] == 5.0
    assert d3['c'] == 5.0
    d3 += {'c': 1.0}
    assert d3['a'] == 1.0
    assert d3['b'] == 5.0
    assert d3['c'] == 6.0
    d4 = {'a': 7.0} + d1
    assert d4['a'] == 8.0
    d5 = d1 + {'a': 9.0}
    assert d5['a'] == 10.0
    d6 = d1 - d2
    assert d6 == {'a': 1.0, 'b': -1.0, 'c': -5.0}
    d6 -= d3
    assert d6 == {'b': -6.0, 'c': -11.0}