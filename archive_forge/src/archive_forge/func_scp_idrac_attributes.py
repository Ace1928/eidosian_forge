from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def scp_idrac_attributes(module, idrac, res_id):
    job_wait = module.params.get('job_wait', True)
    idrac_attr = module.params.get('idrac_attributes')
    system_attr = module.params.get('system_attributes')
    lc_attr = module.params.get('lifecycle_controller_attributes')
    root = '<SystemConfiguration>{0}</SystemConfiguration>'
    component = ''
    idrac_json_data, system_json_data, lc_json_data = ({}, {}, {})
    if idrac_attr is not None:
        idrac_xml_payload, idrac_json_data = xml_data_conversion(idrac_attr, fqdd=MANAGER_ID)
        component += idrac_xml_payload
    if system_attr is not None:
        system_xml_payload, system_json_data = xml_data_conversion(system_attr, fqdd=SYSTEM_ID)
        component += system_xml_payload
    if lc_attr is not None:
        lc_xml_payload, lc_json_data = xml_data_conversion(lc_attr, fqdd=LC_ID)
        component += lc_xml_payload
    get_check_mode(module, idrac, idrac_json_data, system_json_data, lc_json_data)
    payload = root.format(component)
    resp = idrac.import_scp(import_buffer=payload, target='ALL', job_wait=False)
    job_id = resp.headers['Location'].split('/')[-1]
    job_uri = JOB_URI.format(manager_id=res_id, job_id=job_id)
    job_resp = idrac.wait_for_job_completion(job_uri, job_wait=job_wait)
    return job_resp