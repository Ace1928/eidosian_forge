from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_baseline_id(module, baseline_name, rest_obj):
    report = rest_obj.get_all_report_details(BASELINE_URI)
    base_id, template_id = (None, None)
    for base in report['report_list']:
        if base['Name'] == baseline_name:
            base_id = base['Id']
            template_id = base['TemplateId']
            break
    else:
        module.fail_json(msg="Unable to complete the operation because the entered target baseline name '{0}' is invalid.".format(baseline_name))
    return (base_id, template_id)