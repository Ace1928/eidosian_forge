from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_add():
    d1 = ArithmeticDict(int, [('a', 1), ('b', 2)])
    d2 = d1 + 3
    assert d2['a'] == 4
    assert d2['b'] == 5
    d2 = 3 + d1
    assert d2['a'] == 4
    assert d2['b'] == 5
    assert d2['c'] == 0
    d3 = d1 + d2
    assert d3['a'] == 5
    assert d3['b'] == 7
    assert d3['c'] == 0