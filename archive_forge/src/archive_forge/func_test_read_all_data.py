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
def test_read_all_data(self):
    """
        Replicate what goes on in the Swift driver with the
        repeated creation of the ChunkReader object
        """
    CHUNKSIZE = 100
    data = b'*' * units.Ki
    expected_checksum = md5(data, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(data).hexdigest()
    data_file = tempfile.NamedTemporaryFile()
    data_file.write(data)
    data_file.flush()
    infile = open(data_file.name, 'rb')
    bytes_read = 0
    checksum = md5(usedforsecurity=False)
    os_hash_value = hashlib.sha256()
    while True:
        cr = swift.ChunkReader(infile, checksum, os_hash_value, CHUNKSIZE)
        chunk = cr.read(CHUNKSIZE)
        if len(chunk) == 0:
            self.assertEqual(True, cr.is_zero_size)
            break
        bytes_read += len(chunk)
    self.assertEqual(units.Ki, bytes_read)
    self.assertEqual(expected_checksum, cr.checksum.hexdigest())
    self.assertEqual(expected_multihash, cr.os_hash_value.hexdigest())
    data_file.close()
    infile.close()