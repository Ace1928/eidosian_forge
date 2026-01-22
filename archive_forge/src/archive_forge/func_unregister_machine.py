import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def unregister_machine(self, nics, uuid, wait=None, timeout=600):
    """Unregister Baremetal from Ironic

        Removes entries for Network Interfaces and baremetal nodes
        from an Ironic API

        :param nics: An array of strings that consist of MAC addresses
            to be removed.
        :param string uuid: The UUID of the node to be deleted.
        :param wait: DEPRECATED, do not use.
        :param timeout: Integer value, representing seconds with a default
            value of 600, which controls the maximum amount of time to block
            until a lock is released on machine.

        :raises: :class:`~openstack.exceptions.SDKException` on operation
            failure.
        """
    if wait is not None:
        warnings.warn('wait argument is deprecated and has no effect', os_warnings.OpenStackDeprecationWarning)
    machine = self.get_machine(uuid)
    invalid_states = ['active', 'cleaning', 'clean wait', 'clean failed']
    if machine['provision_state'] in invalid_states:
        raise exceptions.SDKException("Error unregistering node '%s' due to current provision state '%s'" % (uuid, machine['provision_state']))
    try:
        self.baremetal.wait_for_node_reservation(machine, timeout)
    except exceptions.SDKException as e:
        raise exceptions.SDKException("Error unregistering node '%s': Exception occured while waiting to be able to proceed: %s" % (machine['uuid'], e))
    for nic in _normalize_port_list(nics):
        try:
            port = next(self.baremetal.ports(address=nic['address']))
        except StopIteration:
            continue
        self.baremetal.delete_port(port.id)
    self.baremetal.delete_node(uuid)