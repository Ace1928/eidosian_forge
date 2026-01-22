from openstack.identity.v2 import _proxy
from openstack.identity.v2 import role
from openstack.identity.v2 import tenant
from openstack.identity.v2 import user
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_role_find(self):
    self.verify_find(self.proxy.find_role, role.Role)