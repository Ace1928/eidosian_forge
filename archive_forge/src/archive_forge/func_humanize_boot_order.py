from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_vm_by_id, wait_for_task, TaskError
@staticmethod
def humanize_boot_order(boot_order):
    results = []
    for device in boot_order:
        if isinstance(device, vim.vm.BootOptions.BootableCdromDevice):
            results.append('cdrom')
        elif isinstance(device, vim.vm.BootOptions.BootableDiskDevice):
            results.append('disk')
        elif isinstance(device, vim.vm.BootOptions.BootableEthernetDevice):
            results.append('ethernet')
        elif isinstance(device, vim.vm.BootOptions.BootableFloppyDevice):
            results.append('floppy')
    return results