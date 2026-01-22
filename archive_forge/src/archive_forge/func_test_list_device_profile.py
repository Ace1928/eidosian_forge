from openstack.accelerator.v2 import _proxy
from openstack.accelerator.v2 import accelerator_request
from openstack.accelerator.v2 import deployable
from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_list_device_profile(self):
    self.verify_list(self.proxy.device_profiles, device_profile.DeviceProfile)