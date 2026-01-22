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
def get_category_payload(module, rest_obj):
    inp_catalog_list = module.params.get('category')
    cdict_ref = get_category_data_tree(rest_obj)
    if not cdict_ref:
        module.exit_json(failed=True, msg=CATEGORY_FETCH_FAILED)
    payload_cat_list = get_category_payloadlist(module, inp_catalog_list, cdict_ref)
    return payload_cat_list