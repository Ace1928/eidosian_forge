from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_rl(module, array, local_rl):
    """Create Pod Replica Link"""
    changed = False
    if module.params['pause'] is not None:
        if local_rl['status'] != 'paused' and module.params['pause']:
            changed = True
            if not module.check_mode:
                try:
                    array.pause_pod_replica_link(local_pod_name=module.params['name'], remote_pod_name=local_rl['remote_pod_name'])
                except Exception:
                    module.fail_json(msg='Failed to pause replica link {0}.'.format(module.params['name']))
        elif local_rl['status'] == 'paused' and (not module.params['pause']):
            changed = True
            if not module.check_mode:
                try:
                    array.resume_pod_replica_link(local_pod_name=module.params['name'], remote_pod_name=local_rl['remote_pod_name'])
                except Exception:
                    module.fail_json(msg='Failed to resume replica link {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)