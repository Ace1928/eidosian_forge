from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def update_storage_systems_info(self):
    """Get current web services proxy storage systems."""
    try:
        rc, existing_systems = self.request('storage-systems')
        for system in self.systems:
            for existing_system in existing_systems:
                if system['ssid'] == existing_system['id']:
                    system['current_info'] = existing_system
                    if system['current_info']['passwordStatus'] in ['unknown', 'securityLockout']:
                        system['failed'] = True
                        self.module.warn('Skipping storage system [%s] because of current password status [%s]' % (system['ssid'], system['current_info']['passwordStatus']))
                    if system['current_info']['metaTags']:
                        system['current_info']['metaTags'] = sorted(system['current_info']['metaTags'], key=lambda x: x['key'])
                    break
            else:
                self.systems_to_add.append(system)
        for existing_system in existing_systems:
            for system in self.systems:
                if existing_system['id'] == system['ssid']:
                    if existing_system['id'] in self.undiscovered_systems:
                        self.undiscovered_systems.remove(existing_system['id'])
                        self.module.warn('Expected storage system exists on the proxy but was failed to be discovered. Array [%s].' % existing_system['id'])
                    break
            else:
                self.systems_to_remove.append(existing_system['id'])
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve storage systems. Error [%s].' % to_native(error))