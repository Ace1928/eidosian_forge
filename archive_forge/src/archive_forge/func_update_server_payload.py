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
def update_server_payload(module, payload):
    ipv4 = {'enable_dhcp': 'enableDHCPIPv4', 'enable_ipv4': 'enableIPv4', 'static_alternate_dns_server': 'staticAlternateDNSIPv4', 'static_gateway': 'staticGatewayIPv4', 'static_ip_address': 'staticIPAddressIPv4', 'static_preferred_dns_server': 'staticPreferredDNSIPv4', 'static_subnet_mask': 'staticSubnetMaskIPv4', 'use_dhcp_to_obtain_dns_server_address': 'useDHCPToObtainDNSIPv4'}
    ipv6 = {'enable_auto_configuration': 'enableAutoConfigurationIPv6', 'enable_ipv6': 'enableIPv6', 'static_alternate_dns_server': 'staticAlternateDNSIPv6', 'static_gateway': 'staticGatewayIPv6', 'static_ip_address': 'staticIPAddressIPv6', 'static_preferred_dns_server': 'staticPreferredDNSIPv6', 'static_prefix_length': 'staticPrefixLengthIPv6', 'use_dhcpv6_to_obtain_dns_server_address': 'useDHCPToObtainDNSIPv6'}
    vlan = {'enable_vlan': 'vlanEnable', 'vlan_id': 'vlanId'}
    diff = {}
    mparams = validate_dependency(module.params)
    enable_nic = mparams.get('enable_nic')
    bool_trans = {True: 'Enabled', False: 'Disabled'}
    if enable_nic:
        if mparams.get('ipv4_configuration'):
            df = transform_diff(mparams.get('ipv4_configuration'), ipv4, payload, bool_trans)
            diff.update(df)
        if mparams.get('ipv6_configuration'):
            df = transform_diff(mparams.get('ipv6_configuration'), ipv6, payload, bool_trans)
            diff.update(df)
        if mparams.get('management_vlan'):
            df = transform_diff(mparams.get('management_vlan'), vlan, payload, bool_trans)
            diff.update(df)
    enable_nic = bool_trans.get(enable_nic)
    if payload.get('enableNIC') != enable_nic:
        payload['enableNIC'] = enable_nic
        diff.update({'enableNIC': enable_nic})
    return diff