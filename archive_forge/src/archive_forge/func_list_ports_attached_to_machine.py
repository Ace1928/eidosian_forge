import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def list_ports_attached_to_machine(self, name_or_id):
    """List virtual ports attached to the bare metal machine.

        :param string name_or_id: A machine name or UUID.
        :returns: List of ``openstack.Resource`` objects representing
            the ports.
        """
    machine = self.get_machine(name_or_id)
    vif_ids = self.baremetal.list_node_vifs(machine)
    return [self.get_port(vif) for vif in vif_ids]