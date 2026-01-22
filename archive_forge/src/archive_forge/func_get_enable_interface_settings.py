from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat, pprint
import time
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_enable_interface_settings(self, iface, expected_iface, update, body):
    """Enable or disable the IPv4 network interface."""
    if self.enable_interface:
        if not iface['enabled']:
            update = True
        body['ipv4Enabled'] = True
    else:
        if iface['enabled']:
            update = True
        body['ipv4Enabled'] = False
    expected_iface['enabled'] = body['ipv4Enabled']
    return (update, expected_iface, body)