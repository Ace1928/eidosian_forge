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
def test_delete_object(self):
    self.register_uris([dict(method='HEAD', uri=self.object_endpoint, headers={'X-Object-Meta': 'foo'}), dict(method='DELETE', uri=self.object_endpoint, status_code=204)])
    self.assertTrue(self.cloud.delete_object(self.container, self.object))
    self.assert_calls()