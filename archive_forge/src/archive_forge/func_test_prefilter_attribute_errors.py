import pytest
from IPython.core.prefilter import AutocallChecker
def test_prefilter_attribute_errors():
    """Capture exceptions thrown by user objects on attribute access.

    See http://github.com/ipython/ipython/issues/988."""

    class X(object):

        def __getattr__(self, k):
            raise ValueError('broken object')

        def __call__(self, x):
            return x
    ip.user_ns['x'] = X()
    ip.magic('autocall 2')
    try:
        ip.prefilter('x 1')
    finally:
        del ip.user_ns['x']
        ip.magic('autocall 0')