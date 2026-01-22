from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def get_service_tag_with_fqdn(rest_obj, module):
    """
    get the service tag, if hostname is dnsname
    """
    hostname = module.params['hostname']
    service_tag = None
    device_details = rest_obj.get_all_items_with_pagination(DEVICE_URI)
    for each_device in device_details['value']:
        for item in each_device['DeviceManagement']:
            if item.get('DnsName') == hostname or item.get('NetworkAddress') == hostname:
                return each_device['DeviceServiceTag']
    return service_tag