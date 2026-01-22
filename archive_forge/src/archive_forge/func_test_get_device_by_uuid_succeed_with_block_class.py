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
@mock.patch.object(lightos.LightOSConnector, '_check_device_exists_reading_block_class', return_value='/dev/nvme0n1')
def test_get_device_by_uuid_succeed_with_block_class(self, execute_mock):
    self.assertEqual(self.connector._get_device_by_uuid(FAKE_VOLUME_UUID), '/dev/nvme0n1')