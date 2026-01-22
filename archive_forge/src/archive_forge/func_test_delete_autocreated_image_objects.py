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
def test_delete_autocreated_image_objects(self):
    self.use_keystone_v3()
    self.cloud.image_api_use_tasks = True
    endpoint = self.cloud.object_store.get_endpoint()
    other_image = self.getUniqueString('no-delete')
    self.register_uris([dict(method='GET', uri=self.get_mock_url(service_type='object-store', resource=self.container_name, qs_elements=['format=json']), json=[{'content_type': 'application/octet-stream', 'bytes': 1437258240, 'hash': '249219347276c331b87bf1ac2152d9af', 'last_modified': '2015-02-16T17:50:05.289600', 'name': other_image}, {'content_type': 'application/octet-stream', 'bytes': 1290170880, 'hash': fakes.NO_MD5, 'last_modified': '2015-04-14T18:29:00.502530', 'name': self.image_name}]), dict(method='HEAD', uri=self.get_mock_url(service_type='object-store', resource=self.container_name, append=[other_image]), headers={'X-Timestamp': '1429036140.50253', 'X-Trans-Id': 'txbbb825960a3243b49a36f-005a0dadaedfw1', 'Content-Length': '1290170880', 'Last-Modified': 'Tue, 14 Apr 2015 18:29:01 GMT', 'X-Object-Meta-X-Shade-Sha256': 'does not matter', 'X-Object-Meta-X-Shade-Md5': 'does not matter', 'Date': 'Thu, 16 Nov 2017 15:24:30 GMT', 'Accept-Ranges': 'bytes', 'Content-Type': 'application/octet-stream', 'Etag': '249219347276c331b87bf1ac2152d9af'}), dict(method='HEAD', uri=self.get_mock_url(service_type='object-store', resource=self.container_name, append=[self.image_name]), headers={'X-Timestamp': '1429036140.50253', 'X-Trans-Id': 'txbbb825960a3243b49a36f-005a0dadaedfw1', 'Content-Length': '1290170880', 'Last-Modified': 'Tue, 14 Apr 2015 18:29:01 GMT', 'X-Object-Meta-X-Shade-Sha256': fakes.NO_SHA256, 'X-Object-Meta-X-Shade-Md5': fakes.NO_MD5, 'Date': 'Thu, 16 Nov 2017 15:24:30 GMT', 'Accept-Ranges': 'bytes', 'Content-Type': 'application/octet-stream', 'X-Object-Meta-' + self.cloud._OBJECT_AUTOCREATE_KEY: 'true', 'Etag': fakes.NO_MD5, 'X-Static-Large-Object': 'false'}), dict(method='DELETE', uri='{endpoint}/{container}/{object}'.format(endpoint=endpoint, container=self.container_name, object=self.image_name))])
    deleted = self.cloud.delete_autocreated_image_objects(container=self.container_name)
    self.assertTrue(deleted)
    self.assert_calls()