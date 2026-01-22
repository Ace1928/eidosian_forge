import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def list_machines(self):
    """List Machines.

        :returns: list of :class:`~openstack.baremetal.v1.node.Node`.
        """
    return list(self.baremetal.nodes())