from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def check_similar_job(rest_obj, payload):
    query_param = {'$filter': 'JobType/Id eq {0}'.format(payload['JobType'])}
    job_resp = rest_obj.invoke_request('GET', JOBS_URI, query_param=query_param)
    job_list = job_resp.json_data.get('value', [])
    for jb in job_list:
        if jb['JobName'] == payload['JobName'] and jb['JobDescription'] == payload['JobDescription'] and (jb['Schedule'] == payload['Schedule']):
            jb_prm = dict(((k.get('Key'), k.get('Value')) for k in jb.get('Params')))
            if not jb_prm == payload.get('Params'):
                continue
            trgts = dict(((t.get('Id'), t.get('TargetType').get('Name')) for t in jb.get('Targets')))
            if not trgts == payload.get('Targets'):
                continue
            return jb
    return {}