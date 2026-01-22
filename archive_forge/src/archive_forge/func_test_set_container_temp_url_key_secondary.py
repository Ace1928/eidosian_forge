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
def test_set_container_temp_url_key_secondary(self):
    key = 'super-secure-key'
    self.register_uris([dict(method='POST', uri=self.container_endpoint, status_code=204, validate=dict(headers={'x-container-meta-temp-url-key-2': key})), dict(method='HEAD', uri=self.container_endpoint, headers={'x-container-meta-temp-url-key-2': key})])
    self.cloud.object_store.set_container_temp_url_key(self.container, key, secondary=True)
    self.assert_calls()