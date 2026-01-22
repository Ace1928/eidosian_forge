from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def wait_for_introspection(self, introspection, timeout=None, ignore_error=False):
    """Wait for the introspection to finish.

        :param introspection: The value can be the name or ID of an
            introspection (matching bare metal node name or ID) or
            an :class:`~.introspection.Introspection` instance.
        :param timeout: How much (in seconds) to wait for the introspection.
            The value of ``None`` (the default) means no client-side timeout.
        :param ignore_error: If ``True``, this call will raise an exception
            if the introspection reaches the ``error`` state. Otherwise the
            error state is considered successful and the call returns.
        :returns: :class:`~.introspection.Introspection` instance.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if
            introspection fails and ``ignore_error`` is ``False``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
    res = self._get_resource(_introspect.Introspection, introspection)
    return res.wait(self, timeout=timeout, ignore_error=ignore_error)