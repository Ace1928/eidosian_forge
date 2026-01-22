from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_sub():
    d1 = ArithmeticDict(int, [('a', 1), ('b', 2)])
    d2 = d1 - 7
    assert d2['a'] == -6
    assert d2['b'] == -5
    d2 = 3 - d1
    assert d2['a'] == 2
    assert d2['b'] == 1
    assert d2['c'] == 0
    d3 = d1 - d2
    assert d3['a'] == -1
    assert d3['b'] == 1