from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def customize_exist_vm(self):
    task = None
    network_changes = False
    for nw in self.params['networks']:
        for key in nw:
            if key not in ('device_type', 'mac', 'name', 'vlan', 'type', 'start_connected', 'dvswitch_name'):
                network_changes = True
                break
    if any((v is not None for v in self.params['customization'].values())) or network_changes or self.params.get('customization_spec'):
        self.customize_vm(vm_obj=self.current_vm_obj)
    try:
        task = self.current_vm_obj.CustomizeVM_Task(self.customspec)
    except vim.fault.CustomizationFault as e:
        self.module.fail_json(msg='Failed to customization virtual machine due to CustomizationFault: %s' % to_native(e.msg))
    except vim.fault.RuntimeFault as e:
        self.module.fail_json(msg='failed to customization virtual machine due to RuntimeFault: %s' % to_native(e.msg))
    except Exception as e:
        self.module.fail_json(msg='failed to customization virtual machine due to fault: %s' % to_native(e.msg))
    self.wait_for_task(task)
    if task.info.state == 'error':
        return {'changed': self.change_applied, 'failed': True, 'msg': task.info.error.msg, 'op': 'customize_exist'}
    if self.params['wait_for_customization']:
        set_vm_power_state(self.content, self.current_vm_obj, 'poweredon', force=False)
        is_customization_ok = self.wait_for_customization(vm=self.current_vm_obj, timeout=self.params['wait_for_customization_timeout'])
        if not is_customization_ok:
            return {'changed': self.change_applied, 'failed': True, 'msg': 'Customization failed. For detailed information see warnings', 'op': 'wait_for_customize_exist'}
    return {'changed': self.change_applied, 'failed': False}