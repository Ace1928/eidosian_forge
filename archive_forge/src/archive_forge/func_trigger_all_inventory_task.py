from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def trigger_all_inventory_task(rest_obj):
    job_type = {'Id': 8, 'Name': 'Inventory_Task'}
    job_name = 'Refresh Inventory All Devices'
    job_description = REFRESH_JOB_DESC
    target_param = [{'Id': 500, 'Data': 'All-Devices', 'TargetType': {'Id': 6000, 'Name': 'GROUP'}}]
    job_params = [{'Key': 'defaultInventoryTask', 'Value': 'TRUE'}]
    job_resp = rest_obj.job_submission(job_name, job_description, target_param, job_params, job_type)
    job_id = job_resp.json_data.get('Id')
    return job_id