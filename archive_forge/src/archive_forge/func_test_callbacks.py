from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def test_callbacks():

    def check(caller, func, user_data):
        caller = CALLERS[caller]
        func = FUNCS[func]()
        user_data = USER_DATAS[user_data]()
        if func is callback_python:

            def func2(x):
                return func(x, 2.0)
        else:
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)
        assert_equal(caller(func, 1.0), 2.0)
        assert_raises(ValueError, caller, func, ERROR_VALUE)
        assert_equal(caller(func2, 1.0), 3.0)
    for caller in sorted(CALLERS.keys()):
        for func in sorted(FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)