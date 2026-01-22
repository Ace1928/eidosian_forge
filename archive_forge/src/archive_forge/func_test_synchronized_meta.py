from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils.threading, 'RLock')
def test_synchronized_meta(self, mock_rlock_cls):
    fake_cls = type('fake_cls', (object,), dict(method1=lambda x: None, method2=lambda y: None))
    fake_cls = six.add_metaclass(baseutils.SynchronizedMeta)(fake_cls)
    fake_cls().method1()
    fake_cls().method2()
    mock_rlock_cls.assert_called_once_with()
    self.assertEqual(2, mock_rlock_cls.return_value.__exit__.call_count)