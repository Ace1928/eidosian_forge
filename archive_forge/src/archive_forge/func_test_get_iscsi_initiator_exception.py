import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@mock.patch('socket.getfqdn')
def test_get_iscsi_initiator_exception(self, mock_get_fqdn):
    fake_fqdn = 'fakehost.FAKE-DOMAIN.com'
    fake_exc = exceptions.ISCSIInitiatorAPIException(message='fake_message', error_code=1, func_name='fake_func')
    self._mock_run.side_effect = fake_exc
    mock_get_fqdn.return_value = fake_fqdn
    resulted_iqn = self._initiator.get_iscsi_initiator()
    expected_iqn = '%s:%s' % (self._initiator._MS_IQN_PREFIX, fake_fqdn.lower())
    self.assertEqual(expected_iqn, resulted_iqn)