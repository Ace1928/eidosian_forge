from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def update_system_changes(self, system):
    """Determine whether storage system configuration changes are required """
    if system['current_info']:
        system['changes'] = dict()
        if sorted(system['controller_addresses']) != sorted(system['current_info']['managementPaths']) or system['current_info']['ip1'] not in system['current_info']['managementPaths'] or system['current_info']['ip2'] not in system['current_info']['managementPaths']:
            system['changes'].update({'controllerAddresses': system['controller_addresses']})
        if len(system['meta_tags']) != len(system['current_info']['metaTags']):
            if len(system['meta_tags']) == 0:
                system['changes'].update({'removeAllTags': True})
            else:
                system['changes'].update({'metaTags': system['meta_tags']})
        else:
            for index in range(len(system['meta_tags'])):
                if system['current_info']['metaTags'][index]['key'] != system['meta_tags'][index]['key'] or sorted(system['current_info']['metaTags'][index]['valueList']) != sorted(system['meta_tags'][index]['valueList']):
                    system['changes'].update({'metaTags': system['meta_tags']})
                    break
        if system['accept_certificate'] and (not all([controller['certificateStatus'] == 'trusted' for controller in system['current_info']['controllers']])):
            system['changes'].update({'acceptCertificate': True})
    if system['id'] not in self.undiscovered_systems and system['changes']:
        self.systems_to_update.append(system)