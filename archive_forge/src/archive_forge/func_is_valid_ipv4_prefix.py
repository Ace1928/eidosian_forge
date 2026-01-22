import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_ipv4_prefix(ipv4_prefix):
    """Returns True if *ipv4_prefix* is a valid prefix with mask.

    Samples:
        - valid prefix: 1.1.1.0/32, 244.244.244.1/10
        - invalid prefix: 255.2.2.2/2, 2.2.2/22, etc.
    """
    if not isinstance(ipv4_prefix, str):
        return False
    tokens = ipv4_prefix.split('/')
    if len(tokens) != 2:
        return False
    return is_valid_ipv4(tokens[0]) and is_valid_ip_prefix(tokens[1], 32)