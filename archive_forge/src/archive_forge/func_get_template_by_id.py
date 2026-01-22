from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_template_by_id(module, rest_obj, template_id):
    path = TEMPLATE_PATH.format(template_id=template_id)
    template_req = rest_obj.invoke_request('GET', path)
    if template_req.success:
        return template_req.json_data
    else:
        fail_module(module, msg='Unable to complete the operation because the requested template is not present.')