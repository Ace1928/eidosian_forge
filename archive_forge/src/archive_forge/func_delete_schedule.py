from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_schedule(module, array):
    """Delete, ie. disable, Protection Group Schedules"""
    changed = False
    try:
        current_state = array.get_pgroup(module.params['name'], schedule=True)
        if module.params['schedule'] == 'replication':
            if current_state['replicate_enabled']:
                changed = True
                if not module.check_mode:
                    array.set_pgroup(module.params['name'], replicate_enabled=False)
                    array.set_pgroup(module.params['name'], target_days=0, target_per_day=0, target_all_for=1)
                    array.set_pgroup(module.params['name'], replicate_frequency=14400, replicate_blackout=None)
        elif current_state['snap_enabled']:
            changed = True
            if not module.check_mode:
                array.set_pgroup(module.params['name'], snap_enabled=False)
                array.set_pgroup(module.params['name'], days=0, per_day=0, all_for=1)
                array.set_pgroup(module.params['name'], snap_frequency=300)
    except Exception:
        module.fail_json(msg='Deleting pgroup {0} {1} schedule failed.'.format(module.params['name'], module.params['schedule']))
    module.exit_json(changed=changed)