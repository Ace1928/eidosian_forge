from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat, pprint
import time
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_interface_settings(self, iface, expected_iface, update, body):
    """Update network interface settings."""
    if self.config_method == 'dhcp':
        if iface['config_method'] != 'configDhcp':
            update = True
        body['ipv4AddressConfigMethod'] = 'configDhcp'
    else:
        if iface['config_method'] != 'configStatic':
            update = True
        body['ipv4AddressConfigMethod'] = 'configStatic'
        if iface['address'] != self.address:
            update = True
        body['ipv4Address'] = self.address
        if iface['subnet_mask'] != self.subnet_mask:
            update = True
        body['ipv4SubnetMask'] = self.subnet_mask
        if self.gateway and iface['gateway'] != self.gateway:
            update = True
        body['ipv4GatewayAddress'] = self.gateway
        expected_iface['address'] = body['ipv4Address']
        expected_iface['subnet_mask'] = body['ipv4SubnetMask']
        expected_iface['gateway'] = body['ipv4GatewayAddress']
    expected_iface['config_method'] = body['ipv4AddressConfigMethod']
    return (update, expected_iface, body)