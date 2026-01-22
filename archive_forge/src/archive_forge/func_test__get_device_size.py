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
def test__get_device_size(self):
    fake_data = b'fake binary data'
    fake_len = int(math.ceil(float(len(fake_data)) / units.Gi))
    fake_file = io.BytesIO(fake_data)
    dev_size = scaleio.ScaleIOBrickConnector._get_device_size(fake_file)
    self.assertEqual(fake_len, dev_size)