from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_destination_port(self, session):
    if not self.destination_port:
        return
    if self.session_type == 'encapsulatedRemoteMirrorSource':
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(ipAddress=str(self.destination_port))
        session.destinationPort = port
    if self.session_type == 'remoteMirrorSource':
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(uplinkPortName=str(self.destination_port))
        session.destinationPort = port
    if self.session_type == 'remoteMirrorDest':
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(portKey=str(self.destination_port))
        if not self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=port.portKey)):
            self.module.fail_json(msg="Couldn't find port: {0:s}".format(self.destination_port))
        session.destinationPort = port
    if self.session_type == 'dvPortMirror':
        port = vim.dvs.VmwareDistributedVirtualSwitch.VspanPorts(portKey=str(self.destination_port))
        if not self.dv_switch.FetchDVPorts(vim.dvs.PortCriteria(portKey=port.portKey)):
            self.module.fail_json(msg="Couldn't find port: {0:s}".format(self.destination_port))
        session.destinationPort = port