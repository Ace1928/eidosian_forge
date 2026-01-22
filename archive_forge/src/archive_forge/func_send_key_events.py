from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def send_key_events(self, vm_obj, key_queue, sleep_time=0):
    """
        Send USB HID Scan codes individually to prevent dropping or cobblering
        """
    send_keys = 0
    for item in key_queue:
        usb_spec = vim.UsbScanCodeSpec()
        usb_spec.keyEvents.append(item)
        send_keys += vm_obj.PutUsbScanCodes(usb_spec)
        time.sleep(sleep_time)
    return send_keys