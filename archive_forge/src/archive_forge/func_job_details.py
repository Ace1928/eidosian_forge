from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def job_details(rest_obj):
    query_param = {'$filter': 'JobType/Id eq 6'}
    job_resp = rest_obj.invoke_request('GET', JOB_URL, query_param=query_param)
    job_data = job_resp.json_data.get('value')
    tmp_list = [x['Id'] for x in job_data]
    sorted_id = sorted(tmp_list)
    latest_job = [val for val in job_data if val['Id'] == sorted_id[-1]]
    return latest_job[0]