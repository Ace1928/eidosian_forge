from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def build_vm(self, vm):
    return otypes.Vm(comment=vm.get('comment'), memory=convert_to_bytes(vm.get('memory')) if vm.get('memory') else None, memory_policy=otypes.MemoryPolicy(guaranteed=convert_to_bytes(vm.get('memory_guaranteed')), max=convert_to_bytes(vm.get('memory_max'))) if any((vm.get('memory_guaranteed'), vm.get('memory_max'))) else None, initialization=self.get_initialization(vm), display=otypes.Display(smartcard_enabled=vm.get('smartcard_enabled')) if vm.get('smartcard_enabled') is not None else None, sso=otypes.Sso(methods=[otypes.Method(id=otypes.SsoMethod.GUEST_AGENT)] if vm.get('sso') else []) if vm.get('sso') is not None else None, time_zone=otypes.TimeZone(name=vm.get('timezone')) if vm.get('timezone') else None)