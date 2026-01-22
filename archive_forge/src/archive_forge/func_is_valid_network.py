from __future__ import absolute_import, division, print_function
import ipaddress
import re
def is_valid_network(addr):
    """Returns True if `addr` is IPv4 address/submask in bit CIDR notation, False otherwise."""
    match = re.match(_cidr_pattern, addr)
    if match is None:
        return False
    for i in range(4):
        if int(match.group(i + 1)) > 255:
            return False
    mask = int(match.group(5))
    if mask < 8 or mask > 32:
        return False
    return True