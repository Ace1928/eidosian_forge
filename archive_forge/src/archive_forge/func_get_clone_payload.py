from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_clone_payload(module, rest_obj, template_id, view_id):
    attrib_dict = module.params.get('attributes').copy()
    clone_payload = {}
    clone_payload['SourceTemplateId'] = template_id
    clone_payload['NewTemplateName'] = attrib_dict.pop('Name')
    template = get_template_by_name(clone_payload['NewTemplateName'], module, rest_obj)
    if template:
        module.exit_json(msg=TEMPLATE_NAME_EXISTS.format(name=clone_payload['NewTemplateName']))
    clone_payload['ViewTypeId'] = view_id
    if isinstance(attrib_dict, dict):
        clone_payload.update(attrib_dict)
    return clone_payload