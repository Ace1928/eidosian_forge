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
def test_checksum_v2_header(self):
    cache_filter = ChecksumTestCacheFilter()
    headers = {'x-image-meta-checksum': '1234567890', 'Content-MD5': 'abcdefghi'}
    resp = webob.Response(request=self.request, headers=headers)
    cache_filter._process_GET_response(resp, None)
    self.assertEqual('abcdefghi', cache_filter.cache.image_checksum)