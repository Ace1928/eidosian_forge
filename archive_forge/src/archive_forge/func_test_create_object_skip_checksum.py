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
def test_create_object_skip_checksum(self):
    self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': 1000}, slo={'min_segment_size': 500})), dict(method='HEAD', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=200), dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={}))])
    self.cloud.create_object(container=self.container, name=self.object, filename=self.object_file.name, generate_checksums=False)
    self.assert_calls()