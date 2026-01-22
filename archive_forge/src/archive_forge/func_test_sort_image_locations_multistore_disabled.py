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
def test_sort_image_locations_multistore_disabled(self):
    self.config(enabled_backends=None)
    locations = [{'url': 'rbd://aaaaaaaa/images/id', 'metadata': {'store': 'rbd1'}}, {'url': 'rbd://bbbbbbbb/images/id', 'metadata': {'store': 'rbd2'}}, {'url': 'rbd://cccccccc/images/id', 'metadata': {'store': 'rbd3'}}]
    mp = 'glance.common.utils.glance_store.get_store_from_store_identifier'
    with mock.patch(mp) as mock_get_store:
        utils.sort_image_locations(locations)
    self.assertEqual(0, mock_get_store.call_count)