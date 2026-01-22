from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def restore_volume_from_snapshot(module, system):
    """ Use snapshot to restore a volume """
    changed = False
    is_restoring = module.params['restore_volume_from_snapshot']
    volume_type = module.params['volume_type']
    snap_name = module.params['name']
    snap_id = find_vol_id(module, system, snap_name)
    parent_volume_name = module.params['parent_volume_name']
    parent_volume_id = find_vol_id(module, system, parent_volume_name)
    if not is_restoring:
        raise AssertionError('A programming error occurred. is_restoring is not True')
    if volume_type != 'snapshot':
        module.exit_json(msg="Cannot restore a parent volume from snapshot unless the volume type is 'snapshot'")
    if not parent_volume_name:
        module.exit_json(msg='Cannot restore a parent volume from snapshot unless the parent volume name is specified')
    if not module.check_mode:
        restore_url = f'volumes/{parent_volume_id}/restore?approved=true'
        restore_data = {'source_id': snap_id}
        try:
            system.api.post(path=restore_url, data=restore_data)
            changed = True
        except APICommandFailed as err:
            module.fail_json(msg=f'Cannot restore volume {parent_volume_name} from {snap_name}: {err}')
    return changed