from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_interface_from_id_and_name(self, device_id, interface_name):
    """
        Retrieve the interface ID for a device in Cisco Catalyst Center based on device id and interface name.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_id (str): The id of the device.
            interface_name (str): Name of the interface for which details need to be collected.
        Returns:
            str: The interface ID for the specified device and interface name.
        Description:
            The function sends a request to Cisco Catalyst Center to retrieve the interface information
            for the device with the provided device id and interface name and extracts the interface ID from the
            response, and returns the interface ID.
        """
    try:
        interface_detail_params = {'device_id': device_id, 'name': interface_name}
        response = self.dnac._exec(family='devices', function='get_interface_details', params=interface_detail_params)
        self.log("Received API response from 'get_interface_details': {0}".format(str(response)), 'DEBUG')
        response = response.get('response')
        if response:
            self.status = 'success'
            interface_id = response['id']
            self.log('Successfully fetched interface ID ({0}) by using device id {1} and interface name {2}.'.format(interface_id, device_id, interface_name), 'INFO')
            return interface_id
    except Exception as e:
        error_message = 'Error while fetching interface id for interface({0}) from Cisco Catalyst Center: {1}'.format(interface_name, str(e))
        self.log(error_message, 'ERROR')
        self.msg = error_message
        self.status = 'failed'
        return self

    def get_interface_from_ip(self, device_ip):
        """
            Get the interface ID for a device in Cisco Catalyst Center based on its IP address.
            Parameters:
                self (object): An instance of a class used for interacting with Cisco Catalyst Center.
                device_ip (str): The IP address of the device.
            Returns:
                str: The interface ID for the specified device.
            Description:
                The function sends a request to Cisco Catalyst Center to retrieve the interface information
                for the device with the provided IP address and extracts the interface ID from the
                response, and returns the interface ID.
            """
        try:
            response = self.dnac._exec(family='devices', function='get_interface_by_ip', params={'ip_address': device_ip})
            self.log("Received API response from 'get_interface_by_ip': {0}".format(str(response)), 'DEBUG')
            response = response.get('response')
            if response:
                interface_id = response[0]['id']
                self.log("Fetch Interface Id for device '{0}' successfully !!".format(device_ip))
                return interface_id
        except Exception as e:
            error_message = "Error while fetching Interface Id for device '{0}' from Cisco Catalyst Center: {1}".format(device_ip, str(e))
            self.log(error_message, 'ERROR')
            raise Exception(error_message)