from ..arithmeticdict import ArithmeticDict
def test_ArithmeticDict_div_float():
    d = ArithmeticDict(float)
    d['a'] = 6.0
    d['b'] = 9.0
    t = d / 3.0
    assert t['a'] == 2.0
    assert t['b'] == 3.0