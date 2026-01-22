from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_provision_wired_device(self, device_ip):
    """
        Retrieves the provisioning status of a wired device with the specified management IP address in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ip (str): The management IP address of the wired device for which provisioning status is to be retrieved.
        Returns:
            bool: True if the device is provisioned successfully, False otherwise.
        Description:
            This method communicates with Cisco Catalyst Center to check the provisioning status of a wired device.
            It executes the 'get_provisioned_wired_device' API call with the provided device IP address and
            logs the response.
        """
    response = self.dnac._exec(family='sda', function='get_provisioned_wired_device', op_modifies=True, params={'device_management_ip_address': device_ip})
    if response.get('status') == 'failed':
        self.log('Cannot do provisioning for wired device {0} because of {1}.'.format(device_ip, response.get('description')), 'ERROR')
        return False
    return True