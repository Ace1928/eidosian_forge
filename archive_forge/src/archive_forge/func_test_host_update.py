from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
def test_host_update(self):
    self.verify_update(self.proxy.update_host, host.Host, method_kwargs={'segment_id': SEGMENT_ID})