from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def clear_mac_address(self, interface_id, deploy_mode, interface_name):
    """
        Clear the MAC address table on a specific interface of a device.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            interface_id (str): The UUID of the interface where the MAC addresses will be cleared.
            deploy_mode (str): The deployment mode of the device.
            interface_name(str): The name of the interface for which the MAC addresses will be cleared.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function clears the MAC address table on a specific interface of a device.
            The 'deploy_mode' parameter specifies the deployment mode of the device.
            If the operation is successful, the function returns the response from the API call.
            If an error occurs during the operation, the function logs the error details and updates the status accordingly.
        """
    try:
        payload = {'operation': 'ClearMacAddress', 'payload': {}}
        clear_mac_address_payload = {'payload': payload, 'interface_uuid': interface_id, 'deployment_mode': deploy_mode}
        response = self.dnac._exec(family='devices', function='clear_mac_address_table', op_modifies=True, params=clear_mac_address_payload)
        self.log("Received API response from 'clear_mac_address_table': {0}".format(str(response)), 'DEBUG')
        if not (response and isinstance(response, dict)):
            self.status = 'failed'
            self.msg = "Received an empty response from the API 'clear_mac_address_table'. This indicates a failure to clear\n                    the Mac address table for the interface '{0}'".format(interface_name)
            self.log(self.msg, 'ERROR')
            self.result['response'] = self.msg
            return self
        task_id = response.get('response').get('taskId')
        while True:
            execution_details = self.get_task_details(task_id)
            if execution_details.get('isError'):
                self.status = 'failed'
                failure_reason = execution_details.get('failureReason')
                if failure_reason:
                    self.msg = "Failed to clear the Mac address table for the interface '{0}' due to {1}".format(interface_name, failure_reason)
                else:
                    self.msg = "Failed to clear the Mac address table for the interface '{0}'".format(interface_name)
                self.log(self.msg, 'ERROR')
                self.result['response'] = self.msg
                break
            elif 'clear mac address-table' in execution_details.get('data'):
                self.status = 'success'
                self.result['changed'] = True
                self.result['response'] = execution_details
                self.msg = "Successfully executed the task of clearing the Mac address table for interface '{0}'".format(interface_name)
                self.log(self.msg, 'INFO')
                break
    except Exception as e:
        error_msg = 'An exception occurred during the process of clearing the MAC address table for interface {0}, due to -\n                {1}'.format(interface_name, str(e))
        self.log(error_msg, 'WARNING')
        self.result['changed'] = False
        self.result['response'] = error_msg
    return self