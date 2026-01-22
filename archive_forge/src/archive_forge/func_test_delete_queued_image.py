from glance.api.middleware import cache_manage
from glance.api.v2 import cached_images
import glance.common.config
import glance.common.wsgi
import glance.image_cache
from glance.tests import utils as test_utils
from unittest import mock
import webob
@mock.patch.object(cached_images.CacheController, 'delete_queued_image')
def test_delete_queued_image(self, mock_delete_queued_image):
    mock_delete_queued_image.return_value = self.stub_value
    request = webob.Request.blank('/v2/queued_images/' + self.image_id, environ={'REQUEST_METHOD': 'DELETE'})
    resource = self.cache_manage_filter.process_request(request)
    mock_delete_queued_image.assert_called_with(request, image_id=self.image_id)
    self.assertEqual('"' + self.stub_value + '"', resource.body.decode('utf-8'))