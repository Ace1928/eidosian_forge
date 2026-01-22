from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def create_serial_port(self, backing):
    """
        Create a new serial port
        """
    serial_spec = vim.vm.device.VirtualDeviceSpec()
    serial_port = vim.vm.device.VirtualSerialPort()
    serial_port.yieldOnPoll = backing['yield_on_poll']
    backing_type = backing.get('type', backing.get('backing_type', None))
    serial_port.backing = self.get_backing_info(serial_port, backing, backing_type)
    serial_spec.device = serial_port
    return serial_spec