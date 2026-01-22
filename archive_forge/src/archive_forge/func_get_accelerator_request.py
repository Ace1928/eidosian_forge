from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def get_accelerator_request(self, uuid, fields=None):
    """Get a single accelerator request.

        :param uuid: The value can be the UUID of a accelerator request.
        :returns: One :class:
            `~openstack.accelerator.v2.accelerator_request.AcceleratorRequest`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            accelerator request matching the criteria could be found.
        """
    return self._get(_arq.AcceleratorRequest, uuid)