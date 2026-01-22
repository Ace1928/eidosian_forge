import glob
import http.client
import queue
from unittest import mock
from unittest.mock import mock_open
from os_brick import exception
from os_brick.initiator.connectors import lightos
from os_brick.initiator import linuxscsi
from os_brick.privileged import lightos as priv_lightos
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(glob, 'glob', return_value=['/path/nvme0n1/wwid'])
@mock.patch('builtins.open', new_callable=mock_open, read_data=f'uuid.{FAKE_VOLUME_UUID}\n')
def test_check_device_exists_reading_block_class(self, mock_glob, m_open):
    found_dev = self.connector._check_device_exists_reading_block_class(FAKE_VOLUME_UUID)
    self.assertEqual('/dev/nvme0n1', found_dev)