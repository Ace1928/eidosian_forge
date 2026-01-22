import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_cooperative_reader(self):
    """Ensure cooperative reader class accesses all bytes of file"""
    BYTES = 1024
    bytes_read = 0
    with tempfile.TemporaryFile('w+') as tmp_fd:
        tmp_fd.write('*' * BYTES)
        tmp_fd.seek(0)
        for chunk in utils.CooperativeReader(tmp_fd):
            bytes_read += len(chunk)
    self.assertEqual(BYTES, bytes_read)
    bytes_read = 0
    with tempfile.TemporaryFile('w+') as tmp_fd:
        tmp_fd.write('*' * BYTES)
        tmp_fd.seek(0)
        reader = utils.CooperativeReader(tmp_fd)
        byte = reader.read(1)
        while len(byte) != 0:
            bytes_read += 1
            byte = reader.read(1)
    self.assertEqual(BYTES, bytes_read)