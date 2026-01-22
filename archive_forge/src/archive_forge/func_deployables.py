from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def deployables(self, **query):
    """Retrieve a generator of deployables.

        :param kwargs query: Optional query parameters to be sent to
            restrict the deployables to be returned.
        :returns: A generator of deployable instances.
        """
    return self._list(_deployable.Deployable, **query)