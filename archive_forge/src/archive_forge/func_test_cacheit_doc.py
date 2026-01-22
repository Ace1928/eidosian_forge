import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
def test_cacheit_doc():

    @cacheit
    def testfn():
        """test docstring"""
        pass
    assert testfn.__doc__ == 'test docstring'
    assert testfn.__name__ == 'testfn'