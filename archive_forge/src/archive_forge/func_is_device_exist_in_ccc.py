from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def is_device_exist_in_ccc(self, device_ip):
    """
        Check if a device with the given IP exists in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ip (str): The IP address of the device to check.
        Returns:
            bool: True if the device exists, False otherwise.
        Description:
            This method queries Cisco Catalyst Center to check if a device with the specified
            management IP address exists. If the device exists, it returns True; otherwise,
            it returns False. If an error occurs during the process, it logs an error message
            and raises an exception.
        """
    try:
        response = self.dnac._exec(family='devices', function='get_device_list', params={'managementIpAddress': device_ip})
        response = response.get('response')
        if not response:
            self.log("Device with given IP '{0}' is not present in Cisco Catalyst Center".format(device_ip), 'INFO')
            return False
        return True
    except Exception as e:
        error_message = "Error while getting the response of device '{0}' from Cisco Catalyst Center: {1}".format(device_ip, str(e))
        self.log(error_message, 'ERROR')
        raise Exception(error_message)