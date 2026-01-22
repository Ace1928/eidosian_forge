from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ipaddress import ip_interface, ip_network
def is_valid_ip_network(address):
    try:
        ip_network(u'{0}'.format(address))
        return True
    except ValueError:
        return False