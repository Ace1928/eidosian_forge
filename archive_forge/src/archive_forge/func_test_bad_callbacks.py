from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def test_bad_callbacks():

    def check(caller, func, user_data):
        caller = CALLERS[caller]
        user_data = USER_DATAS[user_data]()
        func = BAD_FUNCS[func]()
        if func is callback_python:

            def func2(x):
                return func(x, 2.0)
        else:
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)
        assert_raises(ValueError, caller, LowLevelCallable(func), 1.0)
        assert_raises(ValueError, caller, func2, 1.0)
        llfunc = LowLevelCallable(func)
        try:
            caller(llfunc, 1.0)
        except ValueError as err:
            msg = str(err)
            assert_(llfunc.signature in msg, msg)
            assert_('double (double, double, int *, void *)' in msg, msg)
    for caller in sorted(CALLERS.keys()):
        for func in sorted(BAD_FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)