import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('glob.glob', return_value=[])
def test_get_fc_hbas_no_hbas(self, mock_glob):
    hbas = self.lfc.get_fc_hbas()
    self.assertListEqual([], hbas)
    mock_glob.assert_called_once_with('/sys/class/fc_host/*')