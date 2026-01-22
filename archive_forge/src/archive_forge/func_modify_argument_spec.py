from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def modify_argument_spec(schema):
    if not isinstance(schema, dict):
        return schema
    new_schema = {}
    rename_arg = {'message': 'fmgr_message', 'syslog-facility': 'fmgr_syslog_facility', '80211d': 'd80211d', '80211k': 'd80211k', '80211v': 'd80211v'}
    for param_name in schema:
        if param_name != 'v_range' and param_name != 'api_name':
            new_content = modify_argument_spec(schema[param_name])
            aliase_name = _get_modified_name(param_name)
            if param_name in rename_arg:
                new_content['removed_in_version'] = '3.0.0'
                new_content['removed_from_collection'] = 'fortinet.fortimanager'
                new_content['aliases'] = [rename_arg[param_name]]
            elif aliase_name != param_name:
                new_content['removed_in_version'] = '3.0.0'
                new_content['removed_from_collection'] = 'fortinet.fortimanager'
                if aliase_name not in new_schema and 'api_name' not in schema[param_name]:
                    new_content['aliases'] = [aliase_name]
            new_schema[param_name] = new_content
    return new_schema