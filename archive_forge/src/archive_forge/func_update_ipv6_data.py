from __future__ import (absolute_import, division, print_function)
import copy
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def update_ipv6_data(req_data, ipv6_enabled, ipv6_enabled_deploy, ipv6_nt_deploy, deploy_options):
    if ipv6_enabled is not None and ipv6_enabled is True or (ipv6_enabled_deploy is not None and ipv6_enabled_deploy is True):
        req_data['ProtocolTypeV6'] = None
        if ipv6_enabled is not None:
            req_data['ProtocolTypeV6'] = str(ipv6_enabled).lower()
        ipv6_network_type = deploy_options.get('ipv6_network_type')
        req_data['NetworkTypeV6'] = ipv6_network_type
        if ipv6_network_type == 'Static' or (ipv6_nt_deploy is not None and ipv6_nt_deploy == 'Static'):
            req_data['PrefixLength'] = deploy_options.get('ipv6_prefix_length')
            if deploy_options.get('ipv6_prefix_length') is not None:
                req_data['PrefixLength'] = str(deploy_options.get('ipv6_prefix_length'))
            req_data['IpV6Gateway'] = deploy_options.get('ipv6_gateway')
    elif ipv6_enabled is not None and ipv6_enabled is False:
        req_data['ProtocolTypeV6'] = str(ipv6_enabled).lower()