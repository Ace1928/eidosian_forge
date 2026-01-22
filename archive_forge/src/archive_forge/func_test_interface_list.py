import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_interface_list(self):
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    ep = s.get_endpoint(service_type='baremetal', interface=['internal', 'public'])
    self.assertEqual(ep, self.TEST_BAREMETAL_INTERNAL)
    ep = s.get_endpoint(service_type='baremetal', interface=['public', 'internal'])
    self.assertEqual(ep, self.TEST_BAREMETAL_INTERNAL)
    ep = s.get_endpoint(service_type='compute', interface=['internal', 'public'])
    self.assertEqual(ep, self.TEST_COMPUTE_INTERNAL)
    ep = s.get_endpoint(service_type='compute', interface=['public', 'internal'])
    self.assertEqual(ep, self.TEST_COMPUTE_PUBLIC)