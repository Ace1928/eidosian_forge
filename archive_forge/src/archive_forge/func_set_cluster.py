from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_cluster(module, state, timeout, force):
    if state == 'online':
        cmd = 'pcs cluster start'
    if state == 'offline':
        cmd = 'pcs cluster stop'
        if force:
            cmd = '%s --force' % cmd
    rc, out, err = module.run_command(cmd)
    if rc == 1:
        module.fail_json(msg='Command execution failed.\nCommand: `%s`\nError: %s' % (cmd, err))
    t = time.time()
    ready = False
    while time.time() < t + timeout:
        cluster_state = get_cluster_status(module)
        if cluster_state == state:
            ready = True
            break
    if not ready:
        module.fail_json(msg='Failed to set the state `%s` on the cluster\n' % state)