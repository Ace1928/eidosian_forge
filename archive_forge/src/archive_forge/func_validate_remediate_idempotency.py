from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def validate_remediate_idempotency(module, rest_obj):
    name = module.params['names'][0]
    baseline_info = get_baseline_compliance_info(rest_obj, name, attribute='Name')
    if not any(baseline_info):
        module.fail_json(msg=BASELINE_CHECK_MODE_NOCHANGE_MSG.format(name=name))
    valid_id_list, device_capability_map = get_device_ids(module, rest_obj)
    compliance_reports = rest_obj.get_all_items_with_pagination(CONFIG_COMPLIANCE_URI.format(baseline_info['Id']))
    device_id_list = module.params.get('device_ids')
    device_service_tags_list = module.params.get('device_service_tags')
    if device_id_list:
        compliance_report_map = dict([(item['Id'], item['ComplianceStatus']) for item in compliance_reports['value']])
        if not any(compliance_report_map):
            module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG)
        invalid_values = list(set(device_id_list) - set(compliance_report_map.keys()))
        if invalid_values:
            module.fail_json(INVALID_COMPLIANCE_IDENTIFIER.format('device_ids', ','.join(map(str, invalid_values)), name))
        report_devices = list(set(device_id_list) & set(compliance_report_map.keys()))
        noncomplaint_devices = [device for device in report_devices if compliance_report_map[device] == 'NONCOMPLIANT' or compliance_report_map[device] == 2]
    elif device_service_tags_list:
        compliance_report_map = dict([(item['ServiceTag'], item['ComplianceStatus']) for item in compliance_reports['value']])
        if not any(compliance_report_map):
            module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG)
        invalid_values = list(set(device_service_tags_list) - set(compliance_report_map.keys()))
        if invalid_values:
            module.fail_json(INVALID_COMPLIANCE_IDENTIFIER.format('device_service_tags', ','.join(map(str, invalid_values)), name))
        report_devices = list(set(device_service_tags_list) & set(compliance_report_map.keys()))
        service_tag_id_map = dict([(item['ServiceTag'], item['Id']) for item in compliance_reports['value']])
        noncomplaint_devices = [service_tag_id_map[device] for device in report_devices if compliance_report_map[device] == 'NONCOMPLIANT' or compliance_report_map[device] == 2]
    else:
        compliance_report_map = dict([(item['Id'], item['ComplianceStatus']) for item in compliance_reports['value']])
        if not any(compliance_report_map):
            module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG)
        noncomplaint_devices = [device for device, compliance_status in compliance_report_map.items() if compliance_status == 'NONCOMPLIANT' or compliance_status == 2]
    if len(noncomplaint_devices) == 0:
        module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG)
    if module.check_mode and noncomplaint_devices:
        module.exit_json(msg=CHECK_MODE_CHANGES_MSG, changed=True)
    return (noncomplaint_devices, baseline_info)