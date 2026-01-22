from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def set_pipe_backing(self, serial_port, backing_info):
    """
        Set the pipe backing params
        """
    required_params = ['pipe_name', 'endpoint']
    if set(required_params).issubset(backing_info.keys()):
        backing = serial_port.PipeBackingInfo()
        backing.pipeName = backing_info['pipe_name']
        backing.endpoint = backing_info['endpoint']
    else:
        self.module.fail_json(msg='Failed to create a new serial port of pipe backing type due to insufficient parameters.' + 'The required parameters are pipe_name and endpoint')
    if 'no_rx_loss' in backing_info.keys() and backing_info['no_rx_loss']:
        backing.noRxLoss = backing_info['no_rx_loss']
    return backing