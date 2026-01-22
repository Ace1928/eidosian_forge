import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_server_external_ipv4(cloud, server):
    """Find an externally routable IP for the server.

    There are 5 different scenarios we have to account for:

    * Cloud has externally routable IP from neutron but neutron APIs don't
      work (only info available is in nova server record) (rackspace)
    * Cloud has externally routable IP from neutron (runabove, ovh)
    * Cloud has externally routable IP from neutron AND supports optional
      private tenant networks (vexxhost, unitedstack)
    * Cloud only has private tenant network provided by neutron and requires
      floating-ip for external routing (dreamhost, hp)
    * Cloud only has private tenant network provided by nova-network and
      requires floating-ip for external routing (auro)

    :param cloud: the cloud we're working with
    :param server: the server dict from which we want to get an IPv4 address
    :return: a string containing the IPv4 address or None
    """
    if not cloud.use_external_network():
        return None
    if server['accessIPv4']:
        return server['accessIPv4']
    ext_nets = cloud.get_external_ipv4_networks()
    for ext_net in ext_nets:
        ext_ip = get_server_ip(server, key_name=ext_net['name'], public=True, cloud_public=not cloud.private)
        if ext_ip is not None:
            return ext_ip
    ext_ip = get_server_ip(server, ext_tag='floating', public=True, cloud_public=not cloud.private)
    if ext_ip is not None:
        return ext_ip
    ext_ip = get_server_ip(server, key_name='public', public=True, cloud_public=not cloud.private)
    if ext_ip is not None:
        return ext_ip
    for interfaces in server['addresses'].values():
        for interface in interfaces:
            try:
                ip = ipaddress.ip_address(interface['addr'])
            except Exception:
                continue
            if ip.version == 4 and (not ip.is_private):
                return str(ip)
    return None