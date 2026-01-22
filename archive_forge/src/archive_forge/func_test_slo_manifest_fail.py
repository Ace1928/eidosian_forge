import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def test_slo_manifest_fail(self):
    """
        Uploading the SLO manifest file should be retried up to 3 times before
        giving up. This test fails all 3 attempts and should verify that we
        delete uploaded segments that begin with the object prefix.
        """
    max_file_size = 25
    min_file_size = 1
    uris_to_mock = [dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404)]
    uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}/{index:0>6}'.format(endpoint=self.endpoint, container=self.container, object=self.object, index=index), status_code=201, headers=dict(Etag='etag{index}'.format(index=index))) for index, offset in enumerate(range(0, len(self.content), max_file_size))])
    uris_to_mock.extend([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256})), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=400, validate=dict(params={'multipart-manifest', 'put'}, headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
    uris_to_mock.extend([dict(method='GET', uri='{endpoint}/images?format=json&prefix={prefix}'.format(endpoint=self.endpoint, prefix=self.object), complete_qs=True, json=[{'content_type': 'application/octet-stream', 'bytes': 1437258240, 'hash': '249219347276c331b87bf1ac2152d9af', 'last_modified': '2015-02-16T17:50:05.289600', 'name': self.object}]), dict(method='HEAD', uri='{endpoint}/images/{object}'.format(endpoint=self.endpoint, object=self.object), headers={'X-Timestamp': '1429036140.50253', 'X-Trans-Id': 'txbbb825960a3243b49a36f-005a0dadaedfw1', 'Content-Length': '1290170880', 'Last-Modified': 'Tue, 14 Apr 2015 18:29:01 GMT', 'X-Object-Meta-x-sdk-autocreated': 'true', 'X-Object-Meta-X-Shade-Sha256': 'does not matter', 'X-Object-Meta-X-Shade-Md5': 'does not matter', 'Date': 'Thu, 16 Nov 2017 15:24:30 GMT', 'Accept-Ranges': 'bytes', 'X-Static-Large-Object': 'false', 'Content-Type': 'application/octet-stream', 'Etag': '249219347276c331b87bf1ac2152d9af'}), dict(method='DELETE', uri='{endpoint}/images/{object}'.format(endpoint=self.endpoint, object=self.object))])
    self.register_uris(uris_to_mock)
    self.cloud.image_api_use_tasks = True
    self.assertRaises(exceptions.SDKException, self.cloud.create_object, container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
    self.assert_calls(stop_after=3)