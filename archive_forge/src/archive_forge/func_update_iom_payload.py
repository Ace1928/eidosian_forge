from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def update_iom_payload(module, payload):
    ipv4 = {'enable_dhcp': 'EnableDHCP', 'enable_ipv4': 'EnableIPv4', 'static_gateway': 'StaticGateway', 'static_ip_address': 'StaticIPAddress', 'static_subnet_mask': 'StaticSubnetMask'}
    ipv6 = {'enable_ipv6': 'EnableIPv6', 'static_gateway': 'StaticGateway', 'static_ip_address': 'StaticIPv6Address', 'static_prefix_length': 'StaticPrefixLength', 'enable_auto_configuration': 'UseDHCPv6'}
    dns = {'preferred_dns_server': 'PrimaryDNS', 'alternate_dns_server1': 'SecondaryDNS', 'alternate_dns_server2': 'TertiaryDNS'}
    vlan = {'enable_vlan': 'EnableMgmtVLANId', 'vlan_id': 'MgmtVLANId'}
    diff = {}
    mparams = validate_dependency(module.params)
    if mparams.get('ipv4_configuration'):
        df = transform_diff(mparams.get('ipv4_configuration'), ipv4, payload.get('IomIPv4Settings'))
        diff.update(df)
    if mparams.get('ipv6_configuration'):
        df = transform_diff(mparams.get('ipv6_configuration'), ipv6, payload.get('IomIPv6Settings'))
        diff.update(df)
    if mparams.get('management_vlan'):
        df = transform_diff(mparams.get('management_vlan'), vlan, payload)
        diff.update(df)
    if mparams.get('dns_server_settings'):
        df = transform_diff(mparams.get('dns_server_settings'), dns, payload.get('IomDNSSettings'))
        dns_iom = payload.get('IomDNSSettings')
        if dns_iom.get('SecondaryDNS') and (not dns_iom.get('PrimaryDNS')):
            module.fail_json(msg=DNS_SETT_ERR1)
        if dns_iom.get('TertiaryDNS') and (not dns_iom.get('PrimaryDNS') or not dns_iom.get('SecondaryDNS')):
            module.fail_json(msg=DNS_SETT_ERR2)
        diff.update(df)
    return diff