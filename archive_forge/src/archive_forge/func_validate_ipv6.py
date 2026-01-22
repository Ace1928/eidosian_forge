from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def validate_ipv6(value, module):
    if value:
        address = value.split('/')
        if len(address) != 2:
            module.fail_json(msg='address format is <ipv6 address>/<mask>, got invalid format {0}'.format(value))
        elif not 0 <= int(address[1]) <= 128:
            module.fail_json(msg='invalid value for mask: {0}, mask should be in range 0-128'.format(address[1]))