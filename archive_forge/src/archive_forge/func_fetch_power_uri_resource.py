from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def fetch_power_uri_resource(module, session_obj):
    try:
        resource_id = module.params.get('resource_id')
        static_resource_id_resource = None
        if resource_id:
            static_resource_id_resource = '{0}{1}{2}'.format(session_obj.root_uri, 'Systems/', resource_id)
        error_message1 = 'The target device does not support the system reset feature using Redfish API.'
        system_uri = '{0}{1}'.format(session_obj.root_uri, 'Systems')
        system_resp = session_obj.invoke_request('GET', system_uri)
        system_members = system_resp.json_data.get('Members')
        if len(system_members) > 1 and static_resource_id_resource is None:
            module.fail_json(msg="Multiple devices exists in the system, but option 'resource_id' is not specified.")
        if system_members:
            resource_id_list = [system_id['@odata.id'] for system_id in system_members if '@odata.id' in system_id]
            system_id_res = static_resource_id_resource or resource_id_list[0]
            if system_id_res in resource_id_list:
                system_id_res_resp = session_obj.invoke_request('GET', system_id_res)
                system_id_res_data = system_id_res_resp.json_data
                action_id_res = system_id_res_data.get('Actions')
                if action_id_res:
                    current_state = system_id_res_data['PowerState']
                    power_uri = action_id_res['#ComputerSystem.Reset']['target']
                    allowable_enums = action_id_res['#ComputerSystem.Reset']['ResetType@Redfish.AllowableValues']
                    powerstate_map.update({'power_uri': power_uri, 'allowable_enums': allowable_enums, 'current_state': current_state})
                else:
                    module.fail_json(msg=error_message1)
            else:
                error_message2 = "Invalid device Id '{0}' is provided".format(resource_id)
                module.fail_json(msg=error_message2)
        else:
            module.fail_json(msg=error_message1)
    except HTTPError as err:
        if err.code in [404, 405]:
            module.fail_json(msg=error_message1, error_info=json.load(err))
        raise err