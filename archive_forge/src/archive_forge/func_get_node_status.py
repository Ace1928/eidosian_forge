from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_node_status(module, node='all'):
    if node == 'all':
        cmd = 'pcs cluster pcsd-status %s' % node
    else:
        cmd = 'pcs cluster pcsd-status'
    rc, out, err = module.run_command(cmd)
    if rc == 1:
        module.fail_json(msg='Command execution failed.\nCommand: `%s`\nError: %s' % (cmd, err))
    status = []
    for o in out.splitlines():
        status.append(o.split(':'))
    return status