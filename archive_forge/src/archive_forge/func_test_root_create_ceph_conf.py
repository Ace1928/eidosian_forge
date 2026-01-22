from unittest import mock
import os_brick.privileged as privsep_brick
import os_brick.privileged.rbd as privsep_rbd
from os_brick.tests import base
@mock.patch.object(privsep_rbd, 'get_rbd_class')
@mock.patch.object(privsep_rbd, 'RBDConnector')
def test_root_create_ceph_conf(self, mock_connector, mock_get_class):
    s = mock.sentinel
    res = privsep_rbd.root_create_ceph_conf(s.monitor_ips, s.monitor_ports, s.cluster_name, s.user, s.keyring)
    mock_get_class.assert_called_once_with()
    mock_connector._create_ceph_conf.assert_called_once_with(s.monitor_ips, s.monitor_ports, s.cluster_name, s.user, s.keyring)
    self.assertIs(mock_connector._create_ceph_conf.return_value, res)