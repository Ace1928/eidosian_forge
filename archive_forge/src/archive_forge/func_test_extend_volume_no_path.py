import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(fibre_channel.FibreChannelConnector, 'get_volume_paths')
def test_extend_volume_no_path(self, mock_volume_paths):
    mock_volume_paths.return_value = []
    volume = {'id': 'fake_uuid'}
    wwn = '1234567890123456'
    connection_info = self.fibrechan_connection(volume, '10.0.2.15:3260', wwn)
    self.assertRaises(exception.VolumePathsNotFound, self.connector.extend_volume, connection_info['data'])