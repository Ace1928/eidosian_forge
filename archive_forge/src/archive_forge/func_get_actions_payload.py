from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def get_actions_payload(module, rest_obj):
    action_payload = {}
    inp_actions = module.params.get('actions')
    if inp_actions:
        ref_actions = get_all_actions(rest_obj)
        inp_dict = {x.get('action_name'): {y.get('name'): y.get('value') for y in x.get('parameters', [])} for x in inp_actions}
        if 'Ignore' in inp_dict:
            action_payload['Ignore'] = {'TemplateId': ref_actions.get('Ignore').get('Id'), 'Name': 'Ignore', 'ParameterDetails': {}}
        else:
            for inp_k, inp_val in inp_dict.items():
                if inp_k in ref_actions:
                    action_payload[inp_k] = create_action_payload(inp_k, inp_val, ref_actions, module)
                else:
                    module.exit_json(failed=True, msg=ACTION_DIS_EXIST.format(inp_k))
    return {'Actions': action_payload} if action_payload else {}