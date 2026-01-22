from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def wait_for_allocation(self, allocation, timeout=None, ignore_error=False):
    """Wait for the allocation to become active.

        :param allocation: The value can be the name or ID of an allocation or
            a :class:`~openstack.baremetal.v1.allocation.Allocation` instance.
        :param timeout: How much (in seconds) to wait for the allocation.
            The value of ``None`` (the default) means no client-side timeout.
        :param ignore_error: If ``True``, this call will raise an exception
            if the allocation reaches the ``error`` state. Otherwise the error
            state is considered successful and the call returns.

        :returns: The instance of the allocation.
        :rtype: :class:`~openstack.baremetal.v1.allocation.Allocation`.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if allocation
            fails and ``ignore_error`` is ``False``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
    res = self._get_resource(_allocation.Allocation, allocation)
    return res.wait(self, timeout=timeout, ignore_error=ignore_error)