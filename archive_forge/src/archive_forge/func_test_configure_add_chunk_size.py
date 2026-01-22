import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_configure_add_chunk_size(self):
    chunk_size = units.Gi
    self.config(filesystem_store_chunk_size=chunk_size, group='glance_store')
    self.store.configure_add()
    self.assertEqual(chunk_size, self.store.chunk_size)
    self.assertEqual(chunk_size, self.store.READ_CHUNKSIZE)
    self.assertEqual(chunk_size, self.store.WRITE_CHUNKSIZE)