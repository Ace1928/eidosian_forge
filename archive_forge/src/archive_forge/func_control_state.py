from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def control_state(vm, vms_service, module):
    if vm is None:
        return
    force = module.params['force']
    state = module.params['state']
    vm_service = vms_service.vm_service(vm.id)
    if vm.status == otypes.VmStatus.IMAGE_LOCKED:
        wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.DOWN)
    elif vm.status == otypes.VmStatus.SAVING_STATE:
        wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.SUSPENDED)
    elif vm.status == otypes.VmStatus.UNASSIGNED or vm.status == otypes.VmStatus.UNKNOWN:
        module.fail_json(msg="Not possible to control VM, if it's in '{0}' status".format(vm.status))
    elif vm.status == otypes.VmStatus.POWERING_DOWN:
        if force and state == 'stopped' or state == 'absent':
            vm_service.stop()
            wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.DOWN)
        else:
            wait(service=vm_service, condition=lambda vm: vm.status in [otypes.VmStatus.DOWN, otypes.VmStatus.UP])