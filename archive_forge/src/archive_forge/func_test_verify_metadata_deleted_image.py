import http.client as http
from unittest.mock import patch
from oslo_log.fixture import logging_error as log_fixture
from oslo_policy import policy
from oslo_utils.fixture import uuidsentinel as uuids
import testtools
import webob
import glance.api.middleware.cache
import glance.api.policy
from glance.common import exception
from glance import context
from glance.tests.unit import base
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import test_policy
from glance.tests.unit import utils as unit_test_utils
def test_verify_metadata_deleted_image(self):
    """
        Test verify_metadata raises exception.NotFound for a deleted image
        """
    image_meta = {'status': 'deleted', 'is_public': True, 'deleted': True}
    cache_filter = ProcessRequestTestCacheFilter()
    self.assertRaises(exception.NotFound, cache_filter._verify_metadata, image_meta)