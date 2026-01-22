from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def start_introspection(self, node, manage_boot=None):
    """Create a new introspection from attributes.

        :param node: The value can be either the name or ID of a node or
            a :class:`~openstack.baremetal.v1.node.Node` instance.
        :param bool manage_boot: Whether to manage boot parameters for the
            node. Defaults to the server default (which is `True`).

        :returns: :class:`~.introspection.Introspection` instance.
        """
    node = self._get_resource(_node.Node, node)
    res = _introspect.Introspection.new(connection=self._get_connection(), id=node.id)
    kwargs = {}
    if manage_boot is not None:
        kwargs['manage_boot'] = manage_boot
    return res.create(self, **kwargs)