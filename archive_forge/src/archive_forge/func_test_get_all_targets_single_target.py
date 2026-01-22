from unittest import mock
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator.connectors import fake
from os_brick.tests import base as test_base
def test_get_all_targets_single_target(self):
    connection_properties = {'target_portal': mock.sentinel.target_portal, 'target_iqn': mock.sentinel.target_iqn, 'target_lun': mock.sentinel.target_lun}
    all_targets = self.connector._get_all_targets(connection_properties)
    expected_target = (mock.sentinel.target_portal, mock.sentinel.target_iqn, mock.sentinel.target_lun)
    self.assertEqual([expected_target], all_targets)