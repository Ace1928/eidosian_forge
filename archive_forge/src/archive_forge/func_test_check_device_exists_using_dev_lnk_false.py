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
def test_check_device_exists_using_dev_lnk_false(self):
    self.assertIsNone(self.connector._check_device_exists_using_dev_lnk(FAKE_VOLUME_UUID))