from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def validate_ipaddress(module, ip_type, config, var_list, ip_func):
    ipv_input = module.params.get(config)
    if ipv_input:
        for ipname in var_list:
            val = ipv_input.get(ipname)
            if val and (not ip_func(val)):
                module.fail_json(msg='Invalid {0} address provided for the {1}'.format(ip_type, ipname))