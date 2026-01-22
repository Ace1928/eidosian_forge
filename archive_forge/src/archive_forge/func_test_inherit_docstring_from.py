import sys
import pytest
from numpy.testing import assert_equal, suppress_warnings
from scipy._lib import doccer
@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstrings stripped')
def test_inherit_docstring_from():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)

        class Foo:

            def func(self):
                """Do something useful."""
                return

            def func2(self):
                """Something else."""

        class Bar(Foo):

            @doccer.inherit_docstring_from(Foo)
            def func(self):
                """%(super)sABC"""
                return

            @doccer.inherit_docstring_from(Foo)
            def func2(self):
                return
    assert_equal(Bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(Bar.func2.__doc__, Foo.func2.__doc__)
    bar = Bar()
    assert_equal(bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(bar.func2.__doc__, Foo.func2.__doc__)