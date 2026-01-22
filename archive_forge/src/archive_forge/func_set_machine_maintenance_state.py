import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def set_machine_maintenance_state(self, name_or_id, state=True, reason=None):
    """Set Baremetal Machine Maintenance State

        Sets Baremetal maintenance state and maintenance reason.

        :param string name_or_id: The Name or UUID value representing the
            baremetal node.
        :param boolean state: The desired state of the node. True being in
            maintenance where as False means the machine is not in maintenance
            mode.  This value defaults to True if not explicitly set.
        :param string reason: An optional freeform string that is supplied to
            the baremetal API to allow for notation as to why the node is in
            maintenance state.

        :returns: None
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    if state:
        self.baremetal.set_node_maintenance(name_or_id, reason)
    else:
        self.baremetal.unset_node_maintenance(name_or_id)