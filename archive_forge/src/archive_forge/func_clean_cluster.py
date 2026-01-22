from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def clean_cluster(module, timeout):
    cmd = 'pcs resource cleanup'
    rc, out, err = module.run_command(cmd)
    if rc == 1:
        module.fail_json(msg='Command execution failed.\nCommand: `%s`\nError: %s' % (cmd, err))