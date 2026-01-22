import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_warnings_module():
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})

    def warn_other_module():

        def warn(arr):
            warnings.warn('Some warning 2', stacklevel=2)
            return arr
        np.apply_along_axis(warn, 0, [0])
    assert_warn_len_equal(my_mod, 0)
    with suppress_warnings() as sup:
        sup.record(UserWarning)
        sup.filter(module=np.lib.shape_base)
        warnings.warn('Some warning')
        warn_other_module()
    assert_equal(len(sup.log), 1)
    assert_equal(sup.log[0].message.args[0], 'Some warning')
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    sup.filter(module=my_mod)
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