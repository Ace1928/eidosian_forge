import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def remove_machine_from_maintenance(self, name_or_id):
    """Remove Baremetal Machine from Maintenance State

        Similarly to set_machine_maintenance_state, this method removes a
        machine from maintenance state.  It must be noted that this method
        simpily calls set_machine_maintenace_state for the name_or_id requested
        and sets the state to False.

        :param string name_or_id: The Name or UUID value representing the
            baremetal node.

        :returns: None
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    self.baremetal.unset_node_maintenance(name_or_id)