from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def compliance_report(module, rest_obj):
    baseline_name = module.params.get('baseline')
    device_id = module.params.get('device_id')
    device_service_tag = module.params.get('device_service_tag')
    baseline_id, template_id = get_baseline_id(module, baseline_name, rest_obj)
    report = []
    if device_id:
        compliance_uri = COMPLIANCE_URI.format(baseline_id, device_id)
        baseline_report = rest_obj.invoke_request('GET', compliance_uri)
        if not baseline_report.json_data.get('ComplianceAttributeGroups') and template_id == 0:
            module.fail_json(msg='The compliance report of the device not found as there is no template associated with the baseline.')
        device_compliance = baseline_report.json_data.get('ComplianceAttributeGroups')
    else:
        baseline_report = rest_obj.get_all_items_with_pagination(CONFIG_COMPLIANCE_URI.format(baseline_id))
        if device_service_tag:
            device_id = validate_device(module, baseline_report, device_id=device_id, service_tag=device_service_tag, base_id=baseline_id)
            report = list(filter(lambda d: d['Id'] in [device_id], baseline_report.get('value')))
        else:
            report = baseline_report.get('value')
        device_compliance = report
        if device_compliance:
            for each in device_compliance:
                compliance_uri = COMPLIANCE_URI.format(baseline_id, each['Id'])
                attr_group = rest_obj.invoke_request('GET', compliance_uri)
                each['ComplianceAttributeGroups'] = attr_group.json_data.get('ComplianceAttributeGroups')
    return device_compliance