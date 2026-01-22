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
def get_category_data_tree(rest_obj):
    resp = rest_obj.invoke_request('GET', CATEGORY_URI)
    cat_raw = resp.json_data.get('value', [])
    cat_dict = dict(((category.get('Name'), dict(((y.get('Name'), {y.get('Id'): dict(((z.get('Name'), z.get('Id')) for z in y.get('SubCategoryDetails')))}) for y in category.get('CategoriesDetails')))) for category in cat_raw))
    return cat_dict