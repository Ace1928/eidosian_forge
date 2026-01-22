import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_image_put_v2_import_supported(self):
    self.cloud.image_api_use_tasks = False
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images', self.image_name], base_url_append='v2'), status_code=404), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2', qs_elements=['name=' + self.image_name]), validate=dict(), json={'images': []}), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2', qs_elements=['os_hidden=True']), json={'images': []}), dict(method='POST', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json=self.fake_image_dict, headers={'OpenStack-image-import-methods': IMPORT_METHODS}, validate=dict(json={'container_format': 'bare', 'disk_format': 'qcow2', 'name': self.image_name, 'owner_specified.openstack.md5': self.fake_image_dict['owner_specified.openstack.md5'], 'owner_specified.openstack.object': self.object_name, 'owner_specified.openstack.sha256': self.fake_image_dict['owner_specified.openstack.sha256'], 'visibility': 'private', 'tags': ['tag1', 'tag2']})), dict(method='PUT', uri=self.get_mock_url('image', append=['images', self.image_id, 'file'], base_url_append='v2'), request_headers={'Content-Type': 'application/octet-stream'}), dict(method='GET', uri=self.get_mock_url('image', append=['images', self.fake_image_dict['id']], base_url_append='v2'), json=self.fake_image_dict), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), complete_qs=True, json=self.fake_search_return)])
    self.cloud.create_image(self.image_name, self.imagefile.name, wait=True, timeout=1, tags=['tag1', 'tag2'], is_public=False, validate_checksum=True)
    self.assert_calls()
    self.assertEqual(self.adapter.request_history[7].text.read(), self.output)