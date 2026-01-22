from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def check_interface_details(self, device_ip, interface_name):
    """
        Checks if the interface details for a device in Cisco Catalyst Center match the specified values in the configuration.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ip (str): The management IP address of the device for which interface details are to be checked.
        Returns:
            bool: True if the interface details match the specified values, False otherwise.
        Description:
            This method retrieves the interface details for a device in Cisco Catalyst Center using the 'get_interface_by_ip' API call.
            It then compares the retrieved details with the specified values in the configuration for updating interface details.
            If all specified parameters match the retrieved values or are not provided in the playbook parameters, the function
            returns True, indicating successful validation.
        """
    device_id = self.get_device_ids([device_ip])
    if not device_id:
        self.log("Error: Device with IP '{0}' not found in Cisco Catalyst Center.Unable to update interface details.".format(device_ip), 'ERROR')
        return False
    interface_detail_params = {'device_id': device_id[0], 'name': interface_name}
    response = self.dnac._exec(family='devices', function='get_interface_details', params=interface_detail_params)
    self.log("Received API response from 'get_interface_details': {0}".format(str(response)), 'DEBUG')
    response = response.get('response')
    if not response:
        self.log("No response received from the API 'get_interface_details'.", 'DEBUG')
        return False
    response_params = {'description': response.get('description'), 'adminStatus': response.get('adminStatus'), 'voiceVlanId': response.get('voiceVlan'), 'vlanId': int(response.get('vlanId'))}
    interface_playbook_params = self.config[0].get('update_interface_details')
    playbook_params = {'description': interface_playbook_params.get('description', ''), 'adminStatus': interface_playbook_params.get('admin_status'), 'voiceVlanId': interface_playbook_params.get('voice_vlan_id', ''), 'vlanId': interface_playbook_params.get('vlan_id')}
    for key, value in playbook_params.items():
        if not value:
            continue
        elif response_params[key] != value:
            return False
    return True