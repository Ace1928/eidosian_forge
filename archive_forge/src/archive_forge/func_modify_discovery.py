from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def modify_discovery(module, rest_obj, discov_list):
    if len(discov_list) > 1:
        dup_discovery = list((item['DiscoveryConfigGroupId'] for item in discov_list))
        module.fail_json(msg=MULTI_DISCOVERY, discovery_ids=dup_discovery)
    job_state_dict = get_discovery_states(rest_obj)
    for d in discov_list:
        if job_state_dict.get(d['DiscoveryConfigGroupId']) == 2050:
            module.fail_json(msg=DISC_JOB_RUNNING.format(name=d['DiscoveryConfigGroupName'], id=d['DiscoveryConfigGroupId']))
    discovery_payload = {'DiscoveryConfigModels': get_discovery_config(module, rest_obj), 'Schedule': get_schedule(module)}
    other_params = get_other_discovery_payload(module)
    discovery_payload.update(other_params)
    update_modify_payload(discovery_payload, discov_list[0], module.params.get('new_name'))
    resp = rest_obj.invoke_request('PUT', CONFIG_GROUPS_ID_URI.format(group_id=discovery_payload['DiscoveryConfigGroupId']), data=discovery_payload)
    job_id = get_job_data(resp.json_data, rest_obj)
    exit_discovery(module, rest_obj, job_id)