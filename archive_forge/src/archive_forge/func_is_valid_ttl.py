import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
@staticmethod
def is_valid_ttl(ipvx):
    version = ipvx.version
    if version == 4:
        return ipvx.ttl == VRRP_IPV4_TTL
    if version == 6:
        return ipvx.hop_limit == VRRP_IPV6_HOP_LIMIT
    raise ValueError('invalid ip version %d' % version)