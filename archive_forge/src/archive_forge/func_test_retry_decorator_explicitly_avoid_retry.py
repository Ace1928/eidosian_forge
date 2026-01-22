from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch('time.sleep')
def test_retry_decorator_explicitly_avoid_retry(self, mock_sleep):

    def func_side_effect(fake_arg, retry_context):
        self.assertEqual(mock.sentinel.arg, fake_arg)
        self.assertEqual(retry_context, dict(prevent_retry=False))
        retry_context['prevent_retry'] = True
        raise exceptions.Win32Exception(message='fake_exc', error_code=1)
    fake_func, mock_side_effect = self._get_fake_func_with_retry_decorator(exceptions=exceptions.Win32Exception, side_effect=func_side_effect, pass_retry_context=True)
    self.assertRaises(exceptions.Win32Exception, fake_func, mock.sentinel.arg)
    self.assertEqual(1, mock_side_effect.call_count)
    self.assertFalse(mock_sleep.called)