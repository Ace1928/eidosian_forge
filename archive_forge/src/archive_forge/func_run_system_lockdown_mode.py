from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def run_system_lockdown_mode(idrac, module):
    """
    Get Lifecycle Controller status

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    """
    msg = {'changed': False, 'failed': False, 'msg': 'Successfully completed the lockdown mode operations.'}
    idrac.use_redfish = True
    share_path = tempfile.gettempdir() + os.sep
    upd_share = file_share_manager.create_share_obj(share_path=share_path, isFolder=True)
    if not upd_share.IsValid:
        module.fail_json(msg='Unable to access the share. Ensure that the share name, share mount, and share credentials provided are correct.')
    set_liason = idrac.config_mgr.set_liason_share(upd_share)
    if set_liason['Status'] == 'Failed':
        try:
            message = set_liason['Data']['Message']
        except (IndexError, KeyError):
            message = set_liason['Message']
        module.fail_json(msg=message)
    if module.params['lockdown_mode'] == 'Enabled':
        msg['system_lockdown_status'] = idrac.config_mgr.enable_system_lockdown()
    elif module.params['lockdown_mode'] == 'Disabled':
        msg['system_lockdown_status'] = idrac.config_mgr.disable_system_lockdown()
    if msg.get('system_lockdown_status') and 'Status' in msg['system_lockdown_status']:
        if msg['system_lockdown_status']['Status'] == 'Success':
            msg['changed'] = True
        else:
            module.fail_json(msg='Failed to complete the lockdown mode operations.')
    return msg