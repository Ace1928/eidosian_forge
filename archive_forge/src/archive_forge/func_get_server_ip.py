import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_server_ip(server, public=False, cloud_public=True, **kwargs):
    """Get an IP from the Nova addresses dict

    :param server: The server to pull the address from
    :param public: Whether the address we're looking for should be considered
                   'public' and therefore reachabiliity tests should be
                   used. (defaults to False)
    :param cloud_public: Whether the cloud has been configured to use private
                         IPs from servers as the interface_ip. This inverts the
                         public reachability logic, as in this case it's the
                         private ip we expect shade to be able to reach
    """
    addrs = find_nova_addresses(server['addresses'], **kwargs)
    return find_best_address(addrs, public=public, cloud_public=cloud_public)