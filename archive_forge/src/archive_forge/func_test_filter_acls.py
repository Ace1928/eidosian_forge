from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_filter_acls(self):
    mock_acl = mock.MagicMock()
    mock_acl.Action = self._FAKE_ACL_ACT
    mock_acl.Applicability = self.netutils._ACL_APPLICABILITY_LOCAL
    mock_acl.Direction = self._FAKE_ACL_DIR
    mock_acl.AclType = self._FAKE_ACL_TYPE
    mock_acl.RemoteAddress = self._FAKE_REMOTE_ADDR
    acls = [mock_acl, mock_acl]
    good_acls = self.netutils._filter_acls(acls, self._FAKE_ACL_ACT, self._FAKE_ACL_DIR, self._FAKE_ACL_TYPE, self._FAKE_REMOTE_ADDR)
    bad_acls = self.netutils._filter_acls(acls, self._FAKE_ACL_ACT, self._FAKE_ACL_DIR, self._FAKE_ACL_TYPE)
    self.assertEqual(acls, good_acls)
    self.assertEqual([], bad_acls)