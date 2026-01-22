from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
import ansible_collections.community.general.plugins.module_utils.influxdb as influx
def set_user_grants(module, client, user_name, grants):
    changed = False
    current_grants = []
    try:
        current_grants = client.get_list_privileges(user_name)
    except influx.exceptions.InfluxDBClientError as e:
        if not module.check_mode or 'user not found' not in e.content:
            module.fail_json(msg=e.content)
    try:
        parsed_grants = []
        for i, v in enumerate(current_grants):
            if v['privilege'] != 'NO PRIVILEGES':
                if v['privilege'] == 'ALL PRIVILEGES':
                    v['privilege'] = 'ALL'
                parsed_grants.append(v)
        for current_grant in parsed_grants:
            if current_grant not in grants:
                if not module.check_mode:
                    client.revoke_privilege(current_grant['privilege'], current_grant['database'], user_name)
                changed = True
        for grant in grants:
            if grant not in parsed_grants:
                if not module.check_mode:
                    client.grant_privilege(grant['privilege'], grant['database'], user_name)
                changed = True
    except influx.exceptions.InfluxDBClientError as e:
        module.fail_json(msg=e.content)
    return changed