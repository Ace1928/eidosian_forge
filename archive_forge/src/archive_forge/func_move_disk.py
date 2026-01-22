from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec,
from re import compile, match, sub
from time import sleep
def move_disk(self, disk, vmid, vm, vm_config):
    params = dict()
    params['disk'] = disk
    params['vmid'] = vmid
    params['bwlimit'] = self.module.params['bwlimit']
    params['storage'] = self.module.params['target_storage']
    params['target-disk'] = self.module.params['target_disk']
    params['target-vmid'] = self.module.params['target_vmid']
    params['format'] = self.module.params['format']
    params['delete'] = 1 if self.module.params.get('delete_moved', False) else 0
    params = dict(((k, v) for k, v in params.items() if v is not None))
    if params.get('storage', False):
        disk_config = disk_conf_str_to_dict(vm_config[disk])
        if params['storage'] == disk_config['storage_name']:
            return False
    task_id = self.proxmox_api.nodes(vm['node']).qemu(vmid).move_disk.post(**params)
    task_success = self.wait_till_complete_or_timeout(vm['node'], task_id)
    if task_success:
        return True
    else:
        self.module.fail_json(msg='Reached timeout while waiting for moving VM disk. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(task_id).log.get()[:1])