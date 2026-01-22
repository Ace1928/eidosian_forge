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
def test_get_hash_str(self):
    nfs_conn = nfs.NfsBrickConnector()
    test_str = 'test_str'
    with mock.patch.object(nfs.hashlib, 'sha256') as fake_hashlib:
        nfs_conn.get_hash_str(test_str)
        test_str = test_str.encode('utf-8')
        fake_hashlib.assert_called_once_with(test_str)