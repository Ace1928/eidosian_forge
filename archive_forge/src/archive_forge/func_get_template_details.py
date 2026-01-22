from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_template_details(module, rest_obj):
    id = module.params.get('template_id')
    query_param = {'$filter': 'Id eq {0}'.format(id)}
    srch = 'Id'
    if not id:
        id = module.params.get('template_name')
        query_param = {'$filter': "Name eq '{0}'".format(id)}
        srch = 'Name'
    template = {}
    resp = rest_obj.invoke_request('GET', TEMPLATES_URI, query_param=query_param)
    if resp.success and resp.json_data.get('value'):
        tlist = resp.json_data.get('value', [])
        for xtype in tlist:
            if xtype.get(srch) == id:
                template = xtype
    return template