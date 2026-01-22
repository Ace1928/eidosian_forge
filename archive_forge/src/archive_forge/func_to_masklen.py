from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def to_masklen(val):
    """ converts a netmask to a masklen """
    if not is_netmask(val):
        raise ValueError('invalid value for netmask: %s' % val)
    bits = list()
    for x in val.split('.'):
        octet = bin(int(x)).count('1')
        bits.append(octet)
    return sum(bits)