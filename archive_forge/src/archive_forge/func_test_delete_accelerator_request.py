from openstack.accelerator.v2 import _proxy
from openstack.accelerator.v2 import accelerator_request
from openstack.accelerator.v2 import deployable
from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_delete_accelerator_request(self):
    self.verify_delete(self.proxy.delete_accelerator_request, accelerator_request.AcceleratorRequest, False)