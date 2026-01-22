import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def valid_ipv4(addr, flags=0):
    """
    Wrapper function of "netaddr.valid_ipv4()".

    The function extends "netaddr.valid_ipv4()" to enable to validate
    IPv4 network address in "xxx.xxx.xxx.xxx/xx" format.

    :param addr: IP address to be validated.
    :param flags: See the "netaddr.valid_ipv4()" docs for details.
    :return: True is valid. False otherwise.
    """
    return _valid_ip(netaddr.valid_ipv4, 32, addr, flags)