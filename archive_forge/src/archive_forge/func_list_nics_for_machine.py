import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def list_nics_for_machine(self, uuid):
    """Returns a list of ports present on the machine node.

        :param uuid: String representing machine UUID value in order to
            identify the machine.
        :returns: A list of ports.
        """
    return list(self.baremetal.ports(details=True, node_id=uuid))