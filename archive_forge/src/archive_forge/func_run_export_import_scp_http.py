from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def run_export_import_scp_http(idrac, module):
    share_url = urlparse(module.params['share_name'])
    share = {}
    scp_file = module.params.get('scp_file')
    share['share_ip'] = share_url.netloc
    share['share_name'] = share_url.path.strip('/')
    share['share_type'] = share_url.scheme.upper()
    share['file_name'] = scp_file
    scp_file_name_format = scp_file
    share['username'] = module.params.get('share_user')
    share['password'] = module.params.get('share_password')
    scp_target = ','.join(module.params['scp_components'])
    command = module.params['command']
    if share['share_type'] == 'HTTPS':
        share['ignore_certificate_warning'] = IGNORE_WARNING[module.params['ignore_certificate_warning']]
    if command == 'import':
        perform_check_mode(module, idrac)
        if share['share_type'] in ['HTTP', 'HTTPS']:
            proxy_share = get_proxy_share(module)
            share.update(proxy_share)
        idrac_import_scp_params = {'target': scp_target, 'share': share, 'job_wait': module.params['job_wait'], 'host_powerstate': module.params['end_host_power_state'], 'shutdown_type': module.params['shutdown_type']}
        scp_response = idrac.import_scp_share(**idrac_import_scp_params)
        scp_response = wait_for_job_tracking_redfish(module, idrac, scp_response)
    elif command == 'export':
        scp_file_name_format = get_scp_file_format(module)
        share['file_name'] = scp_file_name_format
        include_in_export = IN_EXPORTS[module.params['include_in_export']]
        if share['share_type'] in ['HTTP', 'HTTPS']:
            proxy_share = get_proxy_share(module)
            share.update(proxy_share)
        scp_response = idrac.export_scp(export_format=module.params['export_format'], export_use=module.params['export_use'], target=scp_target, job_wait=False, share=share, include_in_export=include_in_export)
        scp_response = wait_for_job_tracking_redfish(module, idrac, scp_response)
    scp_response = response_format_change(scp_response, module.params, scp_file_name_format)
    exit_on_failure(module, scp_response, command)
    return scp_response