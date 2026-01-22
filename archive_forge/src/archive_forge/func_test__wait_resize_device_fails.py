import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
@mock.patch.object(time, 'sleep')
def test__wait_resize_device_fails(self, mock_sleep):
    fake_vol = mock.MagicMock()
    fake_vol.size = 2
    fake_file = io.BytesIO(b'fake binary data')
    with mock.patch.object(scaleio.ScaleIOBrickConnector, '_get_device_size', return_value=1):
        self.assertRaises(exceptions.BackendException, scaleio.ScaleIOBrickConnector._wait_resize_device, fake_vol, fake_file)