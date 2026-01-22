from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def get_key_event(self, hid_code, modifiers):
    key_event = vim.UsbScanCodeSpecKeyEvent()
    key_modifier = vim.UsbScanCodeSpecModifierType()
    key_modifier.leftAlt = False
    key_modifier.leftControl = False
    key_modifier.leftGui = False
    key_modifier.leftShift = False
    key_modifier.rightAlt = False
    key_modifier.rightControl = False
    key_modifier.rightGui = False
    key_modifier.rightShift = False
    if 'LEFTSHIFT' in modifiers:
        key_modifier.leftShift = True
    if 'CTRL' in modifiers:
        key_modifier.leftControl = True
    if 'ALT' in modifiers:
        key_modifier.leftAlt = True
    key_event.modifiers = key_modifier
    key_event.usbHidCode = self.hid_to_hex(hid_code)
    return key_event