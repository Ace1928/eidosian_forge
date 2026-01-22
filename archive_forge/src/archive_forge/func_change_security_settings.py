from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (IBMSVCRestApi,
from ansible.module_utils._text import to_native
def change_security_settings(self):
    cmd = 'chsecurity'
    cmd_opts = {}
    for attr, value in vars(self).items():
        if attr in ['restapi', 'log', 'module', 'clustername', 'domain', 'username', 'password', 'validate_certs', 'token', 'log_path']:
            continue
        cmd_opts[attr] = value
    result = self.restapi.svc_run_command(cmd, cmd_opts, cmdargs=None)
    if result == '':
        self.changed = True
        self.log('chsecurity successful !!')
    else:
        self.module.fail_json(msg='chsecurity failed !!')