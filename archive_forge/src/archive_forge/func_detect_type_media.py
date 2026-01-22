from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network
def detect_type_media(self, interfaces):
    for iface in interfaces:
        if 'media' in interfaces[iface]:
            if 'ether' in interfaces[iface]['media'].lower():
                interfaces[iface]['type'] = 'ether'
    return interfaces