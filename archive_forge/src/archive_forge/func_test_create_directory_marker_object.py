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
def test_create_directory_marker_object(self):
    self.register_uris([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object=self.object), status_code=201, validate=dict(headers={'content-type': 'application/directory'}))])
    self.cloud.create_directory_marker_object(container=self.container, name=self.object)
    self.assert_calls()