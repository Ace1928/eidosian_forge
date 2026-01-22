from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def update_deployable(self, uuid, patch):
    """Reconfig the FPGA with new bitstream.

        :param uuid: The value can be the UUID of a deployable
        :param patch: The information to reconfig.
        :returns: The results of FPGA reconfig.
        """
    return self._get_resource(_deployable.Deployable, uuid).patch(self, patch)