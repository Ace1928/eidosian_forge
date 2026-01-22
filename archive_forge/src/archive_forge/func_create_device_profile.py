from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def create_device_profile(self, **attrs):
    """Create a device_profile.

        :param kwargs attrs: a list of device_profiles.
        :returns: The list of created device profiles
        """
    return self._create(_device_profile.DeviceProfile, **attrs)