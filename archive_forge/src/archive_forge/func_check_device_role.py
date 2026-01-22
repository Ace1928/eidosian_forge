from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def check_device_role(self, device_ip):
    """
        Checks if the device role and role source for a device in Cisco Catalyst Center match the specified values in the configuration.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ip (str): The management IP address of the device for which the device role is to be checked.
        Returns:
            bool: True if the device role and role source match the specified values, False otherwise.
        Description:
            This method retrieves the device role and role source for a device in Cisco Catalyst Center using the
            'get_device_response' method and compares the retrieved values with specified values in the configuration
            for updating device roles.
        """
    role = self.config[0].get('role')
    response = self.get_device_response(device_ip)
    return response.get('role') == role