from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def get_all_ips(ip_addresses, module):
    ip_addresses_list = []
    for ip in ip_addresses:
        try:
            if '/' in ip:
                cidr_list = IPNetwork(ip)
                ip_addresses_list.append(cidr_list)
            elif '-' in ip and ip.count('-') == 1:
                range_addr = ip.split('-')
                range_list = IPRange(range_addr[0], range_addr[1])
                ip_addresses_list.append(range_list)
            else:
                single_ip = IPAddress(ip)
                ip_addresses_list.append(single_ip)
        except (AddrFormatError, ValueError):
            module.fail_json(msg=INVALID_IP_FORMAT.format(ip))
    return ip_addresses_list