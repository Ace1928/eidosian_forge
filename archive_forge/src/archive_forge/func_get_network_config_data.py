from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_network_config_data(rest_obj, module):
    try:
        interface = module.params.get('interface_name')
        resp = rest_obj.invoke_request('GET', JOB_IP_CONFIG)
        adapter_list = resp.json_data.get('value')
        int_adp = None
        pri_adp = None
        if adapter_list:
            for adp in adapter_list:
                if interface and adp.get('InterfaceName') == interface:
                    int_adp = adp
                    break
                if adp.get('PrimaryInterface'):
                    pri_adp = adp
        if interface and int_adp is None:
            module.fail_json(msg="The 'interface_name' value provided {0} is invalid".format(interface))
        elif int_adp:
            return (int_adp, 'POST', POST_IP_CONFIG)
        else:
            return (pri_adp, 'POST', POST_IP_CONFIG)
    except HTTPError:
        pass
    except Exception as err:
        raise err
    resp = rest_obj.invoke_request('GET', IP_CONFIG)
    return (resp.json_data, 'PUT', IP_CONFIG)