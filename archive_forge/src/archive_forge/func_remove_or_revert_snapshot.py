from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, list_snapshots, vmware_argument_spec
def remove_or_revert_snapshot(self, vm):
    if vm.snapshot is None:
        vm_name = self.module.params.get('uuid') or self.module.params.get('name')
        if self.module.params.get('state') == 'revert':
            self.module.fail_json(msg='virtual machine - %s does not have any snapshots to revert to.' % vm_name)
        self.module.exit_json(msg="virtual machine - %s doesn't have any snapshots to remove." % vm_name)
    if self.module.params['snapshot_name']:
        snap_obj = self.get_snapshots_by_name_recursively(vm.snapshot.rootSnapshotList, self.module.params['snapshot_name'])
    elif self.module.params['snapshot_id']:
        snap_obj = self.get_snapshots_by_id_recursively(vm.snapshot.rootSnapshotList, self.module.params['snapshot_id'])
    task = None
    if len(snap_obj) == 1:
        snap_obj = snap_obj[0].snapshot
        if self.module.params['state'] == 'absent':
            remove_children = self.module.params.get('remove_children', False)
            task = snap_obj.RemoveSnapshot_Task(remove_children)
        elif self.module.params['state'] == 'revert':
            task = snap_obj.RevertToSnapshot_Task()
    else:
        vm_id = self.module.params.get('uuid') or self.module.params.get('name') or self.params.get('moid')
        self.module.exit_json(msg="Couldn't find any snapshots with specified name: %s on VM: %s" % (self.module.params['snapshot_name'], vm_id))
    return task