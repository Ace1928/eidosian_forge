from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def delete_discovery(module, rest_obj, discov_list):
    job_state_dict = get_discovery_states(rest_obj)
    delete_ids = []
    for d in discov_list:
        if job_state_dict.get(d['DiscoveryConfigGroupId']) == 2050:
            module.fail_json(msg=DISC_JOB_RUNNING.format(name=d['DiscoveryConfigGroupName'], id=d['DiscoveryConfigGroupId']))
        else:
            delete_ids.append(d['DiscoveryConfigGroupId'])
    delete_payload = {'DiscoveryGroupIds': delete_ids}
    rest_obj.invoke_request('POST', DELETE_JOB_URI, data=delete_payload)
    module.exit_json(msg=DISC_DEL_JOBS_SUCCESS.format(n=len(delete_ids)), changed=True)