from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def update_console_preferences(module, rest_obj, payload, payload_cifs, job_payload, job, payload_dict, schedule):
    cifs_resp = None
    job_final_resp = None
    get_bas = module.params.get('builtin_appliance_share')
    device_health = module.params.get('device_health')
    [payload['ConsoleSetting'].remove(i) for i in payload['ConsoleSetting'] if i['Name'] == 'SHARE_TYPE']
    if device_health and device_health.get('health_check_interval_unit') and (job['Schedule'] != schedule):
        job_final_resp = rest_obj.invoke_request('POST', JOB_URL, data=job_payload)
    if get_bas and get_bas.get('share_options') and (payload_dict['SHARE_TYPE']['Value'] != get_bas.get('share_options')):
        cifs_resp = rest_obj.invoke_request('POST', CIFS_URL, data=payload_cifs)
    final_resp = rest_obj.invoke_request('POST', SETTINGS_URL, data=payload)
    return (final_resp, cifs_resp, job_final_resp)