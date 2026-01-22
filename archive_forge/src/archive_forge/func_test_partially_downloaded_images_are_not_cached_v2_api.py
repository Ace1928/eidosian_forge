import http.client as http_client
import os
import shutil
import httplib2
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import units
from glance.tests import functional
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
@skip_if_disabled
def test_partially_downloaded_images_are_not_cached_v2_api(self):
    """
        Verify that we do not cache images that were downloaded partially
        using v2 images API.
        """
    self.cleanup()
    self.start_servers(**self.__dict__.copy())
    path = 'http://%s:%d/v2/images' % ('127.0.0.1', self.api_port)
    http = httplib2.Http()
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    image_entity = {'name': 'Image1', 'visibility': 'public', 'container_format': 'bare', 'disk_format': 'raw'}
    response, content = http.request(path, 'POST', headers=headers, body=jsonutils.dumps(image_entity))
    self.assertEqual(http_client.CREATED, response.status)
    data = jsonutils.loads(content)
    image_id = data['id']
    path = 'http://%s:%d/v2/images/%s/file' % ('127.0.0.1', self.api_port, image_id)
    headers = self._headers({'content-type': 'application/octet-stream'})
    image_data = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    response, content = http.request(path, 'PUT', headers=headers, body=image_data)
    self.assertEqual(http_client.NO_CONTENT, response.status)
    image_cached_path = os.path.join(self.api_server.image_cache_dir, image_id)
    self.assertFalse(os.path.exists(image_cached_path))
    http = httplib2.Http()
    range_ = 'bytes=3-5'
    headers = self._headers({'Range': range_})
    response, content = http.request(path, 'GET', headers=headers)
    self.assertEqual(http_client.PARTIAL_CONTENT, response.status)
    self.assertEqual(b'DEF', content)
    content_range = 'bytes 3-5/*'
    headers = self._headers({'Content-Range': content_range})
    response, content = http.request(path, 'GET', headers=headers)
    self.assertEqual(http_client.PARTIAL_CONTENT, response.status)
    self.assertEqual(b'DEF', content)
    image_cached_path = os.path.join(self.api_server.image_cache_dir, image_id)
    self.assertFalse(os.path.exists(image_cached_path))
    self.stop_servers()