from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_device_response(self, device_ip):
    """
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ip (str): The management IP address of the device for which the response is to be retrieved.
        Returns:
            dict: A dictionary containing details of the device obtained from the Cisco Catalyst Center.
        Description:
            This method communicates with Cisco Catalyst Center to retrieve the details of a device with the specified
            management IP address. It executes the 'get_device_list' API call with the provided device IP address,
            logs the response, and returns a dictionary containing information about the device.
        """
    try:
        response = self.dnac._exec(family='devices', function='get_device_list', params={'managementIpAddress': device_ip})
        response = response.get('response')[0]
    except Exception as e:
        error_message = 'Error while getting the response of device from Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'ERROR')
        raise Exception(error_message)
    return response