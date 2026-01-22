from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network

    This is a generic BSD subclass of Network using the ifconfig command.
    It defines
    - interfaces (a list of interface names)
    - interface_<name> dictionary of ipv4, ipv6, and mac address information.
    - all_ipv4_addresses and all_ipv6_addresses: lists of all configured addresses.
    