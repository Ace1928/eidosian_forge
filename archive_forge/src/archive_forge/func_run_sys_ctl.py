from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def run_sys_ctl(module, args):
    sys_ctl = [module.get_bin_path('system-control', required=True)]
    if module.params['user']:
        sys_ctl = sys_ctl + ['--user']
    return module.run_command(sys_ctl + args)