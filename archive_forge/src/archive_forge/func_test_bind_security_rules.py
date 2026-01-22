from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(networkutils.NetworkUtils, '_create_security_acl')
@mock.patch.object(networkutils.NetworkUtils, '_get_new_weights')
@mock.patch.object(networkutils.NetworkUtils, '_filter_security_acls')
def test_bind_security_rules(self, mock_filtered_acls, mock_get_weights, mock_create_acl, mock_get_elem_assoc_cls):
    m_port = mock.MagicMock()
    m_acl = mock.MagicMock()
    mock_get_elem_assoc_cls.return_value = [m_acl]
    mock_filtered_acls.return_value = []
    mock_get_weights.return_value = [mock.sentinel.FAKE_WEIGHT]
    mock_create_acl.return_value = m_acl
    fake_rule = mock.MagicMock()
    self.netutils._bind_security_rules(m_port, [fake_rule])
    mock_create_acl.assert_called_once_with(fake_rule, mock.sentinel.FAKE_WEIGHT)
    mock_add_features = self.netutils._jobutils.add_multiple_virt_features
    mock_add_features.assert_called_once_with([m_acl], m_port)
    mock_get_elem_assoc_cls.assert_called_once_with(self.netutils._conn, self.netutils._PORT_EXT_ACL_SET_DATA, element_instance_id=m_port.InstanceID)