from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def last_execution_detail_of_a_job(rest_obj, job_id):
    try:
        last_execution_detail = get_uri_detail(rest_obj, LAST_EXECUTION_DETAIL_URI.format(job_id))
    except Exception:
        pass
    return last_execution_detail