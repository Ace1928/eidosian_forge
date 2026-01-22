import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import services
def test_service_list_with_marker_limit(self):
    expect = [('GET', '/v1/services/?limit=2&marker=%s' % SERVICE2['id'], {}, None)]
    self._test_service_list_with_filters(limit=2, marker=SERVICE2['id'], expect=expect)