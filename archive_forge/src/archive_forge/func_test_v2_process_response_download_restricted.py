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
def test_v2_process_response_download_restricted(self):
    """
        Test process_response for v2 api where _member_ role not able to
        download the image with custom property.
        """
    image_id = 'test1'
    extra_properties = {'x_test_key': 'test_1234'}

    def fake_fetch_request_info(*args, **kwargs):
        return ('test1', 'GET', 'v2')

    def fake_get_v2_image_metadata(*args, **kwargs):
        image = test_policy.ImageStub(image_id, extra_properties=extra_properties)
        request.environ['api.cache.image'] = image
        return (image, glance.api.policy.ImageTarget(image))
    cache_filter = ProcessRequestTestCacheFilter()
    cache_filter._fetch_request_info = fake_fetch_request_info
    cache_filter._get_v2_image_metadata = fake_get_v2_image_metadata
    rules = {'restricted': "not ('test_1234':%(x_test_key)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'get_image': ''}
    self.set_policy_rules(rules)
    cache_filter.policy = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    request = webob.Request.blank('/v2/images/test1/file')
    request.context = context.RequestContext(roles=['_member_'])
    resp = webob.Response(request=request)
    self.assertRaises(webob.exc.HTTPForbidden, cache_filter.process_response, resp)