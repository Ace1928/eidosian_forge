from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def serialize_vm(self, vm):
    """
        Convert a VirtualMachine object to dict.

        :param vm: VirtualMachine object
        :return: dict
        """
    result = self.serialize_obj(vm, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)
    result['id'] = vm.id
    result['name'] = vm.name
    result['type'] = vm.type
    result['location'] = vm.location
    result['tags'] = vm.tags
    result['powerstate'] = dict()
    if vm.instance_view:
        result['powerstate'] = next((s.code.replace('PowerState/', '') for s in vm.instance_view.statuses if s.code.startswith('PowerState')), None)
        for s in vm.instance_view.statuses:
            if s.code.lower() == 'osstate/generalized':
                result['powerstate'] = 'generalized'
    for interface in vm.network_profile.network_interfaces:
        int_dict = azure_id_to_dict(interface.id)
        nic = self.get_network_interface(int_dict['resourceGroups'], int_dict['networkInterfaces'])
        for interface_dict in result['network_profile']['network_interfaces']:
            if interface_dict['id'] == interface.id:
                nic_dict = self.serialize_obj(nic, 'NetworkInterface')
                interface_dict['name'] = int_dict['networkInterfaces']
                interface_dict['properties'] = nic_dict
    for interface in result['network_profile']['network_interfaces']:
        for config in interface['properties']['ip_configurations']:
            if config.get('public_ip_address'):
                pipid_dict = azure_id_to_dict(config['public_ip_address']['id'])
                try:
                    pip = self.network_client.public_ip_addresses.get(pipid_dict['resourceGroups'], pipid_dict['publicIPAddresses'])
                except Exception as exc:
                    self.fail('Error fetching public ip {0} - {1}'.format(pipid_dict['publicIPAddresses'], str(exc)))
                pip_dict = self.serialize_obj(pip, 'PublicIPAddress')
                config['public_ip_address']['name'] = pipid_dict['publicIPAddresses']
                config['public_ip_address']['properties'] = pip_dict['ip_configuration']
    self.log(result, pretty_print=True)
    if self.state != 'absent' and (not result['powerstate']):
        self.fail('Failed to determine PowerState of virtual machine {0}'.format(self.name))
    return result