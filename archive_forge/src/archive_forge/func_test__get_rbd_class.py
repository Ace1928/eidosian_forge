from unittest import mock
import os_brick.privileged as privsep_brick
import os_brick.privileged.rbd as privsep_rbd
from os_brick.tests import base
@mock.patch('oslo_utils.importutils.import_class')
def test__get_rbd_class(self, mock_import):
    self.assertIsNone(privsep_rbd.RBDConnector)
    self.assertIs(privsep_rbd._get_rbd_class, privsep_rbd.get_rbd_class)
    self.addCleanup(setattr, privsep_rbd, 'RBDConnector', None)
    self.addCleanup(setattr, privsep_rbd, 'get_rbd_class', privsep_rbd._get_rbd_class)
    privsep_rbd._get_rbd_class()
    mock_import.assert_called_once_with('os_brick.initiator.connectors.rbd.RBDConnector')
    self.assertEqual(mock_import.return_value, privsep_rbd.RBDConnector)
    self.assertIsNot(privsep_rbd._get_rbd_class, privsep_rbd.get_rbd_class)