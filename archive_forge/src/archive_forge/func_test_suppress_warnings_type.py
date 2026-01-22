import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_warnings_type():
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    sup.filter(UserWarning)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    with suppress_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)