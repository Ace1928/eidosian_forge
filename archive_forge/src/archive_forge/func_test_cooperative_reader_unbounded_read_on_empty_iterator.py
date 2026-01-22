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
def test_cooperative_reader_unbounded_read_on_empty_iterator(self):
    """Ensure cooperative reader is happy with empty iterators"""
    reader = utils.CooperativeReader([])
    self.assertEqual(b'', reader.read())