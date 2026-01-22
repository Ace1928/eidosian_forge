from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def device_exists_in_dnac(self):
    """
        Check which devices already exists in Cisco Catalyst Center and return both device_exist and device_not_exist in dnac.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Cisco Catalyst Center.
        Returns:
            list: A list of devices that exist in Cisco Catalyst Center.
        Description:
            Queries Cisco Catalyst Center to check which devices are already present in Cisco Catalyst Center and store
            its management IP address in the list of devices that exist.
        Example:
            To use this method, create an instance of the class and call 'device_exists_in_dnac' on it,
            The method returns a list of management IP addressesfor devices that exist in Cisco Catalyst Center.
        """
    device_in_dnac = []
    try:
        response = self.dnac._exec(family='devices', function='get_device_list')
    except Exception as e:
        error_message = 'Error while fetching device from Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'CRITICAL')
        raise Exception(error_message)
    if response:
        self.log("Received API response from 'get_device_list': {0}".format(str(response)), 'DEBUG')
        response = response.get('response')
        for ip in response:
            device_ip = ip['managementIpAddress']
            device_in_dnac.append(device_ip)
    return device_in_dnac