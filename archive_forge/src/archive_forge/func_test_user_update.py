from openstack.identity.v2 import _proxy
from openstack.identity.v2 import role
from openstack.identity.v2 import tenant
from openstack.identity.v2 import user
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_user_update(self):
    self.verify_update(self.proxy.update_user, user.User)