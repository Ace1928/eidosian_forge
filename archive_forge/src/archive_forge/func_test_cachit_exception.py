import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
def test_cachit_exception():
    a = []

    @cacheit
    def testf(x):
        a.append(0)
        raise TypeError
    raises(TypeError, lambda: testf(1))
    assert len(a) == 1
    a.clear()
    raises(TypeError, lambda: testf([]))
    assert len(a) == 1

    @cacheit
    def testf2(x):
        a.append(0)
        raise TypeError('Error')
    a.clear()
    raises(TypeError, lambda: testf2(1))
    assert len(a) == 1
    a.clear()
    raises(TypeError, lambda: testf2([]))
    assert len(a) == 1