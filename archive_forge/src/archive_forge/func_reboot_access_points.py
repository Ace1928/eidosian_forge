from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def reboot_access_points(self):
    """
        Reboot access points in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of the class with updated result, status, and log.
        Description:
            This function performs a reboot operation on access points in Cisco Catalyst Center based on the provided IP addresses
            in the configuration. It retrieves the AP devices' MAC addresses, calls the reboot access points API, and monitors
            the progress of the reboot operation.
        """
    device_ips = self.get_device_ips_from_config_priority()
    input_device_ips = device_ips.copy()
    if input_device_ips:
        ap_devices = self.get_ap_devices(input_device_ips)
        self.log('AP Devices from the playbook input are: {0}'.format(str(ap_devices)), 'INFO')
        for device_ip in input_device_ips:
            if device_ip not in ap_devices:
                input_device_ips.remove(device_ip)
    if not input_device_ips:
        self.msg = "No AP Devices IP given in the playbook so can't perform reboot operation"
        self.status = 'success'
        self.result['changed'] = False
        self.result['response'] = self.msg
        self.log(self.msg, 'WARNING')
        return self
    ap_mac_address_list = []
    for device_ip in input_device_ips:
        response = self.dnac._exec(family='devices', function='get_device_list', params={'managementIpAddress': device_ip})
        response = response.get('response')
        if not response:
            continue
        response = response[0]
        ap_mac_address = response.get('apEthernetMacAddress')
        if ap_mac_address is not None:
            ap_mac_address_list.append(ap_mac_address)
    if not ap_mac_address_list:
        self.status = 'success'
        self.result['changed'] = False
        self.msg = 'Cannot find the AP devices for rebooting'
        self.result['response'] = self.msg
        self.log(self.msg, 'INFO')
        return self
    reboot_params = {'apMacAddresses': ap_mac_address_list}
    response = self.dnac._exec(family='wireless', function='reboot_access_points', op_modifies=True, params=reboot_params)
    self.log(str(response))
    if response and isinstance(response, dict):
        task_id = response.get('response').get('taskId')
        while True:
            execution_details = self.get_task_details(task_id)
            if 'url' in execution_details.get('progress'):
                self.status = 'success'
                self.result['changed'] = True
                self.result['response'] = execution_details
                self.msg = 'AP Device(s) {0} successfully rebooted!'.format(str(input_device_ips))
                self.log(self.msg, 'INFO')
                break
            elif execution_details.get('isError'):
                self.status = 'failed'
                failure_reason = execution_details.get('failureReason')
                if failure_reason:
                    self.msg = 'AP Device Rebooting get failed because of {0}'.format(failure_reason)
                else:
                    self.msg = 'AP Device Rebooting get failed'
                self.log(self.msg, 'ERROR')
                break
    return self