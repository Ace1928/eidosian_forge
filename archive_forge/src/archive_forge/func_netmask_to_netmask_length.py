from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def netmask_to_netmask_length(ip_address, netmask, module):
    """
    input: ip_address and netmask in dot notation for IPv4, expanded netmask is not supported for IPv6
           netmask as int or a str representaiton of int is also accepted
    output: netmask length as int
    """
    _check_ipv6_has_prefix_length(ip_address, netmask, module)
    return _get_ipv4orv6_network(ip_address, netmask, False, module).prefixlen