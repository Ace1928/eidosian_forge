from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def func_side_effect(fake_arg, retry_context):
    self.assertEqual(mock.sentinel.arg, fake_arg)
    self.assertEqual(retry_context, dict(prevent_retry=False))
    retry_context['prevent_retry'] = True
    raise exceptions.Win32Exception(message='fake_exc', error_code=1)