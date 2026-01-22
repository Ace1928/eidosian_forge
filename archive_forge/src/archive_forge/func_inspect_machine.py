import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def inspect_machine(self, name_or_id, wait=False, timeout=3600):
    """Inspect a Barmetal machine

        Engages the Ironic node inspection behavior in order to collect
        metadata about the baremetal machine.

        :param name_or_id: String representing machine name or UUID value in
            order to identify the machine.

        :param wait: Boolean value controlling if the method is to wait for
            the desired state to be reached or a failure to occur.

        :param timeout: Integer value, defautling to 3600 seconds, for the
            wait state to reach completion.

        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :returns: Current state of the node.
        """
    return_to_available = False
    node = self.baremetal.get_node(name_or_id)
    if node.provision_state == 'available':
        if node.instance_id:
            raise exceptions.SDKException('Refusing to inspect available machine %(node)s which is associated with an instance (instance_uuid %(inst)s)' % {'node': node.id, 'inst': node.instance_id})
        return_to_available = True
        node = self.baremetal.set_node_provision_state(node, 'manage', wait=True, timeout=timeout)
    if node.provision_state not in ('manageable', 'inspect failed'):
        raise exceptions.SDKException("Machine %(node)s must be in 'manageable', 'inspect failed' or 'available' provision state to start inspection, the current state is %(state)s" % {'node': node.id, 'state': node.provision_state})
    node = self.baremetal.set_node_provision_state(node, 'inspect', wait=True, timeout=timeout)
    if return_to_available:
        node = self.baremetal.set_node_provision_state(node, 'provide', wait=True, timeout=timeout)
    return node