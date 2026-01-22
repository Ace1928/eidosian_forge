from __future__ import absolute_import, division, print_function
import ipaddress
import re
def is_address_in_network(addr, network):
    """Returns True if `addr` and `network` are a valid IPv4 address and
    IPv4 network respectively and if `addr` is in `network`, False otherwise."""
    if not is_valid_address(addr) or not is_valid_network(network):
        return False
    parsed_addr = ipaddress.ip_address(addr)
    parsed_net = ipaddress.ip_network(network)
    return parsed_addr in parsed_net