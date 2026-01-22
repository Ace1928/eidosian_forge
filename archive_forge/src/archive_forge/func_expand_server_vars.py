import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def expand_server_vars(cloud, server):
    """Backwards compatibility function."""
    return add_server_interfaces(cloud, server)