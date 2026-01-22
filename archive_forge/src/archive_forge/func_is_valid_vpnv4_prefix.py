import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_vpnv4_prefix(prefix):
    """Returns True if given prefix is a string represent vpnv4 prefix.

    Vpnv4 prefix is made up of RD:Ipv4, where RD is represents route
    distinguisher and Ipv4 represents valid dot-decimal ipv4 notation string.
    """
    if not isinstance(prefix, str):
        return False
    tokens = prefix.split(':', 2)
    if len(tokens) != 3:
        return False
    if not is_valid_route_dist(':'.join([tokens[0], tokens[1]])):
        return False
    return is_valid_ipv4_prefix(tokens[2])