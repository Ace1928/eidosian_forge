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
def test_cooperative_reader_on_iterator_with_buffer(self):
    """Ensure cooperative reader is happy with empty iterators"""
    data_list = [b'abcd', b'efgh']
    reader = utils.CooperativeReader(data_list)
    self.assertEqual(b'ab', reader.read(2))
    self.assertEqual(b'c', reader.read(1))
    self.assertEqual(b'd', reader.read())
    self.assertEqual(b'efgh', reader.read())
    self.assertEqual(b'', reader.read())