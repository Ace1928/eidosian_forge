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
def sanitize_network_params(self):
    """
        Sanitize user provided network provided params

        Returns: A sanitized list of network params, else fails

        """
    network_devices = list()
    for network in self.params['networks']:
        if 'name' not in network and 'vlan' not in network:
            self.module.fail_json(msg='Please specify at least a network name or a VLAN name under VM network list.')
        if 'name' in network and self.cache.get_network(network['name']) is None:
            self.module.fail_json(msg="Network '%(name)s' does not exist." % network)
        elif 'vlan' in network:
            dvps = self.cache.get_all_objs(self.content, [vim.dvs.DistributedVirtualPortgroup])
            for dvp in dvps:
                if hasattr(dvp.config.defaultPortConfig, 'vlan') and isinstance(dvp.config.defaultPortConfig.vlan.vlanId, int) and (str(dvp.config.defaultPortConfig.vlan.vlanId) == str(network['vlan'])):
                    network['name'] = dvp.config.name
                    break
                if 'dvswitch_name' in network and dvp.config.distributedVirtualSwitch.name == network['dvswitch_name'] and (dvp.config.name == network['vlan']):
                    network['name'] = dvp.config.name
                    break
                if dvp.config.name == network['vlan']:
                    network['name'] = dvp.config.name
                    break
            else:
                self.module.fail_json(msg="VLAN '%(vlan)s' does not exist." % network)
        if 'type' in network:
            if network['type'] not in ['dhcp', 'static']:
                self.module.fail_json(msg="Network type '%(type)s' is not a valid parameter. Valid parameters are ['dhcp', 'static']." % network)
            if network['type'] != 'static' and ('ip' in network or 'netmask' in network):
                self.module.fail_json(msg='Static IP information provided for network "%(name)s", but "type" is set to "%(type)s".' % network)
        elif 'ip' in network or 'netmask' in network:
            network['type'] = 'static'
        else:
            network['type'] = 'dhcp'
        if network.get('type') == 'static':
            if 'ip' in network and 'netmask' not in network:
                self.module.fail_json(msg="'netmask' is required if 'ip' is specified under VM network list.")
            if 'ip' not in network and 'netmask' in network:
                self.module.fail_json(msg="'ip' is required if 'netmask' is specified under VM network list.")
        if 'typev6' in network:
            if network['typev6'] not in ['dhcp', 'static']:
                self.module.fail_json(msg="Network type '%(typev6)s' for IPv6 is not a valid parameter. Valid parameters are ['dhcp', 'static']." % network)
                if network['typev6'] != 'static' and ('ipv6' in network or 'netmaskv6' in network):
                    self.module.fail_json(msg='Static IPv6 information provided for network "%(name)s", but "typev6" is set to "%(typev6)s".' % network)
        elif 'ipv6' in network or 'netmaskv6' in network:
            network['typev6'] = 'static'
        else:
            network['typev6'] = 'dhcp'
        if network.get('typev6') == 'static':
            if 'ipv6' in network and 'netmaskv6' not in network:
                self.module.fail_json(msg="'netmaskv6' is required if 'ipv6' is specified under VM network list.")
            if 'ipv6' not in network and 'netmaskv6' in network:
                self.module.fail_json(msg="'ipv6' is required if 'netmaskv6' is specified under VM network list.")
        if 'device_type' in network and network['device_type'] not in self.device_helper.nic_device_type.keys():
            self.module.fail_json(msg="Device type specified '%s' is not valid. Please specify correct device type from ['%s']." % (network['device_type'], "', '".join(self.device_helper.nic_device_type.keys())))
        if 'mac' in network and (not is_mac(network['mac'])):
            self.module.fail_json(msg="Device MAC address '%s' is invalid. Please provide correct MAC address." % network['mac'])
        network_devices.append(network)
    return network_devices