import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def get_nic_by_mac(self, mac):
    """Get bare metal NIC by its hardware address (usually MAC)."""
    results = list(self.baremetal.ports(address=mac, details=True))
    try:
        return results[0]
    except IndexError:
        return None