from __future__ import (absolute_import, division, print_function)
import sys
import socket
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_default_ip(ipaddr):
    """check the default multicast IP address"""
    if not check_ip_addr(ipaddr):
        return False
    if ipaddr.count('.') != 3:
        return False
    ips = ipaddr.split('.')
    if ips[0] != '224' or ips[1] != '0' or ips[2] != '0':
        return False
    if not ips[3].isdigit() or int(ips[3]) < 107 or int(ips[3]) > 250:
        return False
    return True