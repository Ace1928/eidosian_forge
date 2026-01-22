import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(os.path, 'isdir')
def test_get_all_available_volumes_path_not_dir(self, mock_isdir):
    mock_isdir.return_value = False
    expected = []
    actual = self.connector.get_all_available_volumes()
    self.assertCountEqual(expected, actual)