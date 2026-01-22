from __future__ import (absolute_import, division, print_function)
import copy
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def ip_address_field(module, field, deploy_options, slot=False):
    module_params = deploy_options
    if slot:
        module_params = deploy_options
    for val in field:
        field_value = module_params.get(val[0])
        if field_value is not None:
            valid = validate_ip_address(module_params.get(val[0]), val[1])
            if valid is False:
                module.fail_json(msg=IP_FAIL_MSG.format(field_value, val[0]))