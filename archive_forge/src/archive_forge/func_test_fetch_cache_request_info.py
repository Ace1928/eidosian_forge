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
def test_fetch_cache_request_info(self):
    self.request.environ['api.cache.image_id'] = 'asdf'
    self.request.environ['api.cache.method'] = 'GET'
    self.request.environ['api.cache.version'] = 'v2'
    image_id, method, version = self.middleware._fetch_request_info(self.request)
    self.assertEqual('asdf', image_id)
    self.assertEqual('GET', method)
    self.assertEqual('v2', version)