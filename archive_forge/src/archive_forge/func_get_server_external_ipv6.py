import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_server_external_ipv6(server):
    """Get an IPv6 address reachable from outside the cloud.

    This function assumes that if a server has an IPv6 address, that address
    is reachable from outside the cloud.

    :param server: the server from which we want to get an IPv6 address
    :return: a string containing the IPv6 address or None
    """
    if server['accessIPv6']:
        return server['accessIPv6']
    addresses = find_nova_addresses(addresses=server['addresses'], version=6)
    return find_best_address(addresses, public=True)