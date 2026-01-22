from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def test_signature_override():
    caller = _test_ccallback.test_call_simple
    func = _test_ccallback.test_get_plus1_capsule()
    llcallable = LowLevelCallable(func, signature='bad signature')
    assert_equal(llcallable.signature, 'bad signature')
    assert_raises(ValueError, caller, llcallable, 3)
    llcallable = LowLevelCallable(func, signature='double (double, int *, void *)')
    assert_equal(llcallable.signature, 'double (double, int *, void *)')
    assert_equal(caller(llcallable, 3), 4)