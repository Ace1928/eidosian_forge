import warnings
import pytest
from nibabel import pkg_info
from nibabel.deprecated import (
from nibabel.tests.test_deprecator import TestDeprecatorFunc as _TestDF
def test_futurewarning_mixin():

    class C:

        def __init__(self, val):
            self.val = val

        def meth(self):
            return self.val

    class D(FutureWarningMixin, C):
        pass

    class E(FutureWarningMixin, C):
        warn_message = 'Oh no, not this one'
    with warnings.catch_warnings(record=True) as warns:
        c = C(42)
        assert c.meth() == 42
        assert warns == []
        d = D(42)
        assert d.meth() == 42
        warn = warns.pop(0)
        assert warn.category == FutureWarning
        assert str(warn.message) == 'This class will be removed in future versions'
        e = E(42)
        assert e.meth() == 42
        warn = warns.pop(0)
        assert warn.category == FutureWarning
        assert str(warn.message) == 'Oh no, not this one'