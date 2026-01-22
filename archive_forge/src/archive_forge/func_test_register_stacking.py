from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_register_stacking():
    f = Dispatcher('f')

    @f.register(list)
    @f.register(tuple)
    def rev(x):
        return x[::-1]
    assert f((1, 2, 3)) == (3, 2, 1)
    assert f([1, 2, 3]) == [3, 2, 1]
    assert raises(NotImplementedError, lambda: f('hello'))
    assert rev('hello') == 'olleh'