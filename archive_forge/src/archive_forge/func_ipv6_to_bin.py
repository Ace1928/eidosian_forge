import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def ipv6_to_bin(ip):
    """
    Converts human readable IPv6 string to binary representation.
    :param str ip: IPv6 address string
    :return: binary representation of IPv6 address
    """
    return addrconv.ipv6.text_to_bin(ip)