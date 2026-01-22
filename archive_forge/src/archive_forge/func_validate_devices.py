from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def validate_devices(host_service_tag, rest_obj, module):
    """
    validate domain, primary switch and secondary switch devices
    :param host_service_tag: service tag of the hostname provided
    :param rest_obj: session object
    :param module: Ansible module object
    :return: None
    """
    primary = module.params.get('primary_switch_service_tag')
    secondary = module.params.get('secondary_switch_service_tag')
    device_type_map = rest_obj.get_device_type()
    validate_service_tag(host_service_tag, 'hostname', device_type_map, rest_obj, module)
    validate_service_tag(primary, 'primary_switch_service_tag', device_type_map, rest_obj, module)
    validate_service_tag(secondary, 'secondary_switch_service_tag', device_type_map, rest_obj, module)