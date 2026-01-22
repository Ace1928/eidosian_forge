from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def get_argument_spec():
    """
    Returns a dictionary containing the argument spec for the get_argument_spec function.
    The argument spec is a dictionary that defines the parameters and their types and options for the function.
    The dictionary has the following keys:
        - "license_id": A string representing the license ID.
        - "delete": A boolean representing whether to delete the license.
        - "export": A boolean representing whether to export the license.
        - "import": A boolean representing whether to import the license.
        - "share_parameters": A dictionary representing the share parameters.
            - "type": A string representing the share type.
            - "options": A dictionary representing the options for the share parameters.
                - "share_type": A string representing the share type.
                - "file_name": A string representing the file name.
                - "ip_address": A string representing the IP address.
                - "share_name": A string representing the share name.
                - "workgroup": A string representing the workgroup.
                - "username": A string representing the username.
                - "password": A string representing the password.
                - "ignore_certificate_warning": A string representing whether to ignore certificate warnings.
                - "proxy_support": A string representing the proxy support.
                - "proxy_type": A string representing the proxy type.
                - "proxy_server": A string representing the proxy server.
                - "proxy_port": A integer representing the proxy port.
                - "proxy_username": A string representing the proxy username.
                - "proxy_password": A string representing the proxy password.
            - "required_if": A list of lists representing the required conditions for the share parameters.
            - "required_together": A list of lists representing the required conditions for the share parameters.
        - "resource_id": A string representing the resource ID.
    """
    return {'license_id': {'type': 'str', 'aliases': ['entitlement_id']}, 'delete': {'type': 'bool', 'default': False}, 'export': {'type': 'bool', 'default': False}, 'import': {'type': 'bool', 'default': False}, 'share_parameters': {'type': 'dict', 'options': {'share_type': {'type': 'str', 'default': 'local', 'choices': ['local', 'nfs', 'cifs', 'http', 'https']}, 'file_name': {'type': 'str'}, 'ip_address': {'type': 'str'}, 'share_name': {'type': 'str'}, 'workgroup': {'type': 'str'}, 'username': {'type': 'str'}, 'password': {'type': 'str', 'no_log': True}, 'ignore_certificate_warning': {'type': 'str', 'default': 'off', 'choices': ['off', 'on']}, 'proxy_support': {'type': 'str', 'default': 'off', 'choices': ['off', 'default_proxy', 'parameters_proxy']}, 'proxy_type': {'type': 'str', 'default': 'http', 'choices': ['http', 'socks']}, 'proxy_server': {'type': 'str'}, 'proxy_port': {'type': 'int', 'default': 80}, 'proxy_username': {'type': 'str'}, 'proxy_password': {'type': 'str', 'no_log': True}}, 'required_if': [['share_type', 'local', ['share_name']], ['share_type', 'nfs', ['ip_address', 'share_name']], ['share_type', 'cifs', ['ip_address', 'share_name', 'username', 'password']], ['share_type', 'http', ['ip_address', 'share_name']], ['share_type', 'https', ['ip_address', 'share_name']], ['proxy_support', 'parameters_proxy', ['proxy_server']]], 'required_together': [('username', 'password'), ('proxy_username', 'proxy_password')]}, 'resource_id': {'type': 'str'}}