from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def is_mac(mac_address):
    """
    Validate MAC address for given string
    Args:
        mac_address: string to validate as MAC address

    Returns: (Boolean) True if string is valid MAC address, otherwise False
    """
    mac_addr_regex = re.compile('[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$')
    return bool(mac_addr_regex.match(mac_address.lower()))