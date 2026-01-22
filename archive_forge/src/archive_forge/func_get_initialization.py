from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_initialization(self):
    if self._initialization is not None:
        return self._initialization
    sysprep = self.param('sysprep')
    cloud_init = self.param('cloud_init')
    cloud_init_nics = self.param('cloud_init_nics') or []
    if cloud_init is not None:
        cloud_init_nics.append(cloud_init)
    if cloud_init or cloud_init_nics:
        self._initialization = otypes.Initialization(nic_configurations=[otypes.NicConfiguration(boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol').lower()) if nic.get('nic_boot_protocol') else None, ipv6_boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol_v6').lower()) if nic.get('nic_boot_protocol_v6') else None, name=nic.pop('nic_name', None), on_boot=True, ip=otypes.Ip(address=nic.pop('nic_ip_address', None), netmask=nic.pop('nic_netmask', None), gateway=nic.pop('nic_gateway', None), version=otypes.IpVersion('v4')) if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None else None, ipv6=otypes.Ip(address=nic.pop('nic_ip_address_v6', None), netmask=nic.pop('nic_netmask_v6', None), gateway=nic.pop('nic_gateway_v6', None), version=otypes.IpVersion('v6')) if nic.get('nic_gateway_v6') is not None or nic.get('nic_netmask_v6') is not None or nic.get('nic_ip_address_v6') is not None else None) for nic in cloud_init_nics if nic.get('nic_boot_protocol_v6') is not None or nic.get('nic_ip_address_v6') is not None or nic.get('nic_gateway_v6') is not None or (nic.get('nic_netmask_v6') is not None) or (nic.get('nic_gateway') is not None) or (nic.get('nic_netmask') is not None) or (nic.get('nic_ip_address') is not None) or (nic.get('nic_boot_protocol') is not None)] if cloud_init_nics else None, **cloud_init)
    elif sysprep:
        self._initialization = otypes.Initialization(**sysprep)
    return self._initialization