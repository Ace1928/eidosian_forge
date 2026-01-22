from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ipaddress import ip_interface, ip_network
def compress_address(address):
    addr = ip_network(u'{0}'.format(address))
    result = addr.compressed.split('/', maxsplit=1)[0]
    return result