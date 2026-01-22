import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def list_nics(self):
    """Return a list of all bare metal ports."""
    return list(self.baremetal.ports(details=True))