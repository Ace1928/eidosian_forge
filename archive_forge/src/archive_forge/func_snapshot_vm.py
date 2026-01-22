from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, list_snapshots, vmware_argument_spec
def snapshot_vm(self, vm):
    memory_dump = False
    quiesce = False
    if vm.snapshot is not None:
        snap_obj = self.get_snapshots_by_name_recursively(vm.snapshot.rootSnapshotList, self.module.params['snapshot_name'])
        if snap_obj:
            self.module.exit_json(changed=False, msg='Snapshot named [%(snapshot_name)s] already exists and is current.' % self.module.params)
    if vm.capability.quiescedSnapshotsSupported:
        quiesce = self.module.params['quiesce']
    if vm.capability.memorySnapshotsSupported:
        memory_dump = self.module.params['memory_dump']
    task = None
    try:
        task = vm.CreateSnapshot(self.module.params['snapshot_name'], self.module.params['description'], memory_dump, quiesce)
    except vim.fault.RestrictedVersion as exc:
        self.module.fail_json(msg='Failed to take snapshot due to VMware Licence restriction : %s' % to_native(exc.msg))
    except Exception as exc:
        self.module.fail_json(msg='Failed to create snapshot of virtual machine %s due to %s' % (self.module.params['name'], to_native(exc)))
    return task