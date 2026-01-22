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
def test_create_object(self):
    self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': 1000}, slo={'min_segment_size': 500})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=404), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={'x-object-meta-x-sdk-md5': self.md5, 'x-object-meta-x-sdk-sha256': self.sha256}))])
    self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name)
    self.assert_calls()