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
def test_v2_process_request_response_headers(self):

    def dummy_img_iterator():
        for i in range(3):
            yield i
    image_id = 'test1'
    request = webob.Request.blank('/v2/images/test1/file')
    request.context = context.RequestContext()
    image_meta = {'id': image_id, 'name': 'fake_image', 'status': 'active', 'created_at': '', 'min_disk': '10G', 'min_ram': '1024M', 'protected': False, 'locations': '', 'checksum': 'c1234', 'owner': '', 'disk_format': 'raw', 'container_format': 'bare', 'size': '123456789', 'virtual_size': '123456789', 'is_public': 'public', 'deleted': False, 'updated_at': '', 'properties': {}}
    image = ImageStub(image_id, request.context.project_id)
    request.environ['api.cache.image'] = image
    for k, v in image_meta.items():
        setattr(image, k, v)
    cache_filter = ProcessRequestTestCacheFilter()
    response = cache_filter._process_v2_request(request, image_id, dummy_img_iterator, image_meta)
    self.assertEqual('application/octet-stream', response.headers['Content-Type'])
    self.assertEqual('c1234', response.headers['Content-MD5'])
    self.assertEqual('123456789', response.headers['Content-Length'])