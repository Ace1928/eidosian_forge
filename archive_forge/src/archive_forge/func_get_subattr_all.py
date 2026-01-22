from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_subattr_all(attr_dtls, adv_list):
    attr_detailed = {}
    attr_map = {}
    for each in attr_dtls:
        recurse_subattr_list(each.get('SubAttributeGroups'), each.get('DisplayName'), attr_detailed, attr_map, adv_list)
    return (attr_detailed, attr_map)