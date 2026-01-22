from __future__ import absolute_import, division, print_function
import ipaddress
import re
def is_valid_address(addr):
    """Returns True if `addr` is a valid IPv4 address, False otherwise. Does not support
    octal/hex notations."""
    match = re.match(_addr_pattern, addr)
    if match is None:
        return False
    for i in range(4):
        if int(match.group(i + 1)) > 255:
            return False
    return True