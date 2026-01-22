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
def test_object_segment_retry_failure(self):
    max_file_size = 25
    min_file_size = 1
    self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': max_file_size}, slo={'min_segment_size': min_file_size})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404), dict(method='PUT', uri='{endpoint}/{container}/{object}/000000'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000001'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000002'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201), dict(method='PUT', uri='{endpoint}/{container}/{object}/000003'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=501), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201)])
    self.assertRaises(exceptions.SDKException, self.cloud.create_object, container=self.container, name=self.object, filename=self.object_file.name, use_slo=True)
    self.assert_calls(stop_after=3)