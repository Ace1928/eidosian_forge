import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_ipv6(ipv6):
    """Returns True if given `ipv6` is a valid IPv6 address
    """
    return ip.valid_ipv6(ipv6)