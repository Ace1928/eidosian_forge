from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def discover_array(self):
    """Search for array using the world wide identifier."""
    subnet = ipaddress.ip_network(u'%s' % self.subnet_mask)
    try:
        rc, request_id = self.request('discovery', method='POST', data={'startIP': str(subnet[0]), 'endIP': str(subnet[-1]), 'connectionTimeout': self.DEFAULT_CONNECTION_TIMEOUT_SEC})
        discovered_systems = None
        try:
            for iteration in range(self.DEFAULT_DISCOVERY_TIMEOUT_SEC):
                rc, discovered_systems = self.request('discovery?requestId=%s' % request_id['requestId'])
                if not discovered_systems['discoverProcessRunning']:
                    break
                sleep(1)
            else:
                self.module.fail_json(msg='Timeout waiting for array discovery process. Subnet [%s]' % self.subnet_mask)
        except Exception as error:
            self.module.fail_json(msg='Failed to get the discovery results. Error [%s].' % to_native(error))
        if not discovered_systems:
            self.module.warn('Discovery found no systems. IP starting address [%s]. IP ending address: [%s].' % (str(subnet[0]), str(subnet[-1])))
        else:
            if self.add_discovered_systems:
                for discovered_system in discovered_systems['storageSystems']:
                    if discovered_system['serialNumber'] not in self.serial_numbers:
                        self.systems.append({'ssid': discovered_system['serialNumber'], 'serial': discovered_system['serialNumber'], 'password': self.default_password, 'password_valid': None, 'password_set': None, 'stored_password_valid': None, 'meta_tags': self.default_meta_tags, 'controller_addresses': [], 'embedded_available': None, 'accept_certificate': False, 'current_info': {}, 'changes': {}, 'updated_required': False, 'failed': False, 'discovered': False})
            for system in self.systems:
                for discovered_system in discovered_systems['storageSystems']:
                    if system['serial'] == discovered_system['serialNumber'] or (system['controller_addresses'] and all([address in discovered_system['ipAddresses'] for address in system['controller_addresses']])):
                        system['controller_addresses'] = sorted(discovered_system['ipAddresses'])
                        system['embedded_available'] = 'https' in discovered_system['supportedManagementPorts']
                        system['accept_certificate'] = system['embedded_available'] and self.accept_certificate
                        system['discovered'] = True
                        break
                else:
                    self.undiscovered_systems.append(system['ssid'])
    except Exception as error:
        self.module.fail_json(msg='Failed to initiate array discovery. Error [%s].' % to_native(error))