from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_device_ip_from_id(self, device_id):
    """
        Retrieve the management IP address of a device from Cisco Catalyst Center using its ID.
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_id (str): The unique identifier of the device in Cisco Catalyst Center.
        Returns:
            str: The management IP address of the specified device.
        Raises:
            Exception: If there is an error while retrieving the response from Cisco Catalyst Center.
        Description:
            This method queries Cisco Catalyst Center for the device details based on its unique identifier (ID).
            It uses the 'get_device_list' function in the 'devices' family, extracts the management IP address
            from the response, and returns it. If any error occurs during the process, an exception is raised
            with an appropriate error message logged.
        """
    try:
        response = self.dnac._exec(family='devices', function='get_device_list', params={'id': device_id})
        self.log("Received API response from 'get_device_list': {0}".format(str(response)), 'DEBUG')
        response = response.get('response')[0]
        device_ip = response.get('managementIpAddress')
        return device_ip
    except Exception as e:
        error_message = 'Error occurred while getting the response of device from Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'ERROR')
        raise Exception(error_message)