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
def test_delete_image_task(self):
    self.cloud.image_api_use_tasks = True
    endpoint = self.cloud.object_store.get_endpoint()
    object_path = self.fake_image_dict['owner_specified.openstack.object']
    image_no_checksums = self.fake_image_dict.copy()
    del image_no_checksums['owner_specified.openstack.md5']
    del image_no_checksums['owner_specified.openstack.sha256']
    del image_no_checksums['owner_specified.openstack.object']
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json=self.fake_search_return), dict(method='DELETE', uri='https://image.example.com/v2/images/{id}'.format(id=self.image_id)), dict(method='HEAD', uri='{endpoint}/{object}'.format(endpoint=endpoint, object=object_path), headers={'X-Timestamp': '1429036140.50253', 'X-Trans-Id': 'txbbb825960a3243b49a36f-005a0dadaedfw1', 'Content-Length': '1290170880', 'Last-Modified': 'Tue, 14 Apr 2015 18:29:01 GMT', 'X-Object-Meta-X-Sdk-Sha256': self.fake_image_dict['owner_specified.openstack.sha256'], 'X-Object-Meta-X-Sdk-Md5': self.fake_image_dict['owner_specified.openstack.md5'], 'Date': 'Thu, 16 Nov 2017 15:24:30 GMT', 'Accept-Ranges': 'bytes', 'Content-Type': 'application/octet-stream', 'Etag': fakes.NO_MD5}), dict(method='DELETE', uri='{endpoint}/{object}'.format(endpoint=endpoint, object=object_path))])
    self.cloud.delete_image(self.image_id)
    self.assert_calls()