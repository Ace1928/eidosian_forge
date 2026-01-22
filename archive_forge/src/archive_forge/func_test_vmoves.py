from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
def test_vmoves(self):
    self.verify_list(self.proxy.vmoves, vmove.VMove, method_args=[NOTIFICATION_ID], expected_args=[], expected_kwargs={'notification_id': NOTIFICATION_ID})