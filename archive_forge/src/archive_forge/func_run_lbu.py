from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os.path
def run_lbu(*args):
    code, stdout, stderr = module.run_command([module.get_bin_path('lbu', required=True)] + list(args))
    if code:
        module.fail_json(changed=changed, msg=stderr)
    return stdout