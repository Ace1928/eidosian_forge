from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_not_implemented_error():
    f = Dispatcher('f')

    @f.register(float)
    def _(a):
        raise MDNotImplementedError()
    assert raises(NotImplementedError, lambda: f(1.0))