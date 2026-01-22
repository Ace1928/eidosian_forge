from __future__ import absolute_import, division, print_function
import socket
from functools import total_ordering
from itertools import count, groupby
from ansible.module_utils.six import iteritems
def validate_ipv6_addr(address):
    address = address.split('/')[0]
    try:
        socket.inet_pton(socket.AF_INET6, address)
    except socket.error:
        return False
    return True