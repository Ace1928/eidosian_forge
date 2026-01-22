from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_host_vmhba_info(self):
    """Gather vmhba info"""
    hosts_vmhba_info = {}
    for host in self.hosts:
        host_vmhba_info = dict()
        host_st_system = host.configManager.storageSystem
        if host_st_system:
            device_info = host_st_system.storageDeviceInfo
            host_vmhba_info['vmhba_details'] = []
            for hba in device_info.hostBusAdapter:
                hba_info = dict()
                if hba.pci:
                    hba_info['location'] = hba.pci
                    for pci_device in host.hardware.pciDevice:
                        if pci_device.id == hba.pci:
                            hba_info['adapter'] = pci_device.vendorName + ' ' + pci_device.deviceName
                            break
                else:
                    hba_info['location'] = 'PCI'
                hba_info['device'] = hba.device
                hba_type = hba.key.split('.')[-1].split('-')[0]
                if hba_type == 'SerialAttachedHba':
                    hba_info['type'] = 'SAS'
                elif hba_type == 'FibreChannelHba':
                    hba_info['type'] = 'Fibre Channel'
                else:
                    hba_info['type'] = hba_type
                hba_info['bus'] = hba.bus
                hba_info['status'] = hba.status
                hba_info['model'] = hba.model
                hba_info['driver'] = hba.driver
                try:
                    if isinstance(hba, (vim.host.FibreChannelHba, vim.host.FibreChannelOverEthernetHba)):
                        hba_info['node_wwn'] = self.format_number('%X' % hba.nodeWorldWideName)
                    else:
                        hba_info['node_wwn'] = self.format_number(hba.nodeWorldWideName)
                except AttributeError:
                    pass
                try:
                    if isinstance(hba, (vim.host.FibreChannelHba, vim.host.FibreChannelOverEthernetHba)):
                        hba_info['port_wwn'] = self.format_number('%X' % hba.portWorldWideName)
                    else:
                        hba_info['port_wwn'] = self.format_number(hba.portWorldWideName)
                except AttributeError:
                    pass
                try:
                    hba_info['port_type'] = hba.portType
                except AttributeError:
                    pass
                try:
                    hba_info['speed'] = hba.speed
                except AttributeError:
                    pass
                host_vmhba_info['vmhba_details'].append(hba_info)
        hosts_vmhba_info[host.name] = host_vmhba_info
    return hosts_vmhba_info