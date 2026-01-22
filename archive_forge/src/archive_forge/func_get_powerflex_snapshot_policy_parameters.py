from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_powerflex_snapshot_policy_parameters():
    """This method provide parameter required for the snapshot
    policy module on PowerFlex"""
    return dict(snapshot_policy_name=dict(), snapshot_policy_id=dict(), new_name=dict(), access_mode=dict(choices=['READ_WRITE', 'READ_ONLY']), secure_snapshots=dict(type='bool'), auto_snapshot_creation_cadence=dict(type='dict', options=dict(time=dict(type='int'), unit=dict(choices=['Minute', 'Hour', 'Day', 'Week'], default='Minute'))), num_of_retained_snapshots_per_level=dict(type='list', elements='int'), source_volume=dict(type='list', elements='dict', options=dict(id=dict(), name=dict(), auto_snap_removal_action=dict(choices=['Remove', 'Detach']), detach_locked_auto_snapshots=dict(type='bool'), state=dict(default='present', choices=['present', 'absent']))), pause=dict(type='bool'), state=dict(default='present', choices=['present', 'absent']))