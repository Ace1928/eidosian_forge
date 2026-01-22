from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def strip_smart_fabric_info(module, rest_obj, smart_fabric_info):
    for i in range(len(smart_fabric_info)):
        fabrics_details = smart_fabric_info[i]
        fabrics_details = fetch_smart_fabric_link_details(module, rest_obj, fabrics_details)
        fabrics_details = strip_substr_dict(fabrics_details)
        fabrics_details = clean_data(fabrics_details)
        smart_fabric_info[i] = fabrics_details
    return smart_fabric_info