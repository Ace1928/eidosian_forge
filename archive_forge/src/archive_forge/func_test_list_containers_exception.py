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
def test_list_containers_exception(self):
    endpoint = '{endpoint}/'.format(endpoint=self.endpoint)
    self.register_uris([dict(method='GET', uri=endpoint, complete_qs=True, status_code=416)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_containers)
    self.assert_calls()