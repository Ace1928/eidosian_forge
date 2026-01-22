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
def format_payload(policy):
    pdata = policy.get('PolicyData')
    undiscovered = pdata.get('UndiscoveredTargets')
    if undiscovered:
        pdata['UndiscoveredTargets'] = [{'TargetAddress': x} for x in undiscovered]
    actions = pdata.get('Actions')
    if actions:
        for action in actions.values():
            action['ParameterDetails'] = [{'Name': k, 'Value': v} for k, v in action.get('ParameterDetails', {}).items()]
        pdata['Actions'] = list(actions.values())
    catalogs = pdata.get('Catalogs')
    pdata['Catalogs'] = list(catalogs.values())