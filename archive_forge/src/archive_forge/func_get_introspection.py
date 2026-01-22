from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def get_introspection(self, introspection):
    """Get a specific introspection.

        :param introspection: The value can be the name or ID of an
            introspection (matching bare metal node name or ID) or
            an :class:`~.introspection.Introspection` instance.
        :returns: :class:`~.introspection.Introspection` instance.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            introspection matching the name or ID could be found.
        """
    return self._get(_introspect.Introspection, introspection)