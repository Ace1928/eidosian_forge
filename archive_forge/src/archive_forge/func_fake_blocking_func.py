from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def fake_blocking_func(*args, **kwargs):
    self.assertEqual((mock.sentinel.arg,), args)
    self.assertEqual(dict(kwarg=mock.sentinel.kwarg), kwargs)
    return mock.sentinel.ret_val