import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_read_zero_size_data(self):
    """
        Replicate what goes on in the Swift driver with the
        repeated creation of the ChunkReader object
        """
    expected_checksum = md5(b'', usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(b'').hexdigest()
    CHUNKSIZE = 100
    checksum = md5(usedforsecurity=False)
    os_hash_value = hashlib.sha256()
    data_file = tempfile.NamedTemporaryFile()
    infile = open(data_file.name, 'rb')
    bytes_read = 0
    while True:
        cr = swift.ChunkReader(infile, checksum, os_hash_value, CHUNKSIZE)
        chunk = cr.read(CHUNKSIZE)
        if len(chunk) == 0:
            break
        bytes_read += len(chunk)
    self.assertEqual(True, cr.is_zero_size)
    self.assertEqual(0, bytes_read)
    self.assertEqual(expected_checksum, cr.checksum.hexdigest())
    self.assertEqual(expected_multihash, cr.os_hash_value.hexdigest())
    data_file.close()
    infile.close()