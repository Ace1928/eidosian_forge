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
def test_image_meta(self):
    image_meta = {'x-image-meta-size': 'test'}
    image_meta_properties = {'properties': {'test': 'test'}}
    actual = utils.image_meta_to_http_headers(image_meta)
    actual_test2 = utils.image_meta_to_http_headers(image_meta_properties)
    self.assertEqual({'x-image-meta-x-image-meta-size': 'test'}, actual)
    self.assertEqual({'x-image-meta-property-test': 'test'}, actual_test2)