from openstack.network.v2 import network
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def test_router_add_remove_interface(self):
    iface = self.ROT.add_interface(self.user_cloud.network, subnet_id=self.SUB_ID)
    self._verification(iface)
    iface = self.ROT.remove_interface(self.user_cloud.network, subnet_id=self.SUB_ID)
    self._verification(iface)