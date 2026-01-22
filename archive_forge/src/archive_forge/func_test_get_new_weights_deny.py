from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_new_weights_deny(self):
    mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY)
    mockacl1 = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY, Weight=1)
    mockacl2 = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY, Weight=3)
    actual = self.netutils._get_new_weights([mock_rule, mock_rule], [mockacl1, mockacl2])
    self.assertEqual([2, 4], actual)