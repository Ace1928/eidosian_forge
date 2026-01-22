from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_baseline_from_name(rest_obj, baseline):
    resp = rest_obj.get_all_items_with_pagination(BASELINE_URI)
    baselines_list = resp.get('value')
    bsln = baseline
    for d in baselines_list:
        if d['Name'] == baseline.get('Name'):
            bsln = d
            break
    nlist = list(bsln)
    for k in nlist:
        if str(k).lower().startswith('@odata'):
            bsln.pop(k)
    return bsln