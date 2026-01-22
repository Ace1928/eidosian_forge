import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def assert_head_and_get_return_same_response(self, url, expected_status):
    self.get(url, expected_status=expected_status)
    self.head(url, expected_status=expected_status)