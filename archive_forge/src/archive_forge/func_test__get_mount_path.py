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
def test__get_mount_path(self):
    nfs_conn = nfs.NfsBrickConnector(mountpoint_base='fake_mount_path')
    fake_hex = 'fake_hex_digest'
    fake_share = 'fake_share'
    fake_path = 'fake_mount_path'
    expected_path = os.path.join(fake_path, fake_hex)
    with mock.patch.object(nfs.NfsBrickConnector, 'get_hash_str') as fake_hash:
        fake_hash.return_value = fake_hex
        res = nfs_conn._get_mount_path(fake_share, fake_path)
        self.assertEqual(expected_path, res)