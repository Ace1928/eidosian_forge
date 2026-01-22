from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def resync_devices(self):
    """
        Resync devices in Cisco Catalyst Center.
        This function performs the Resync operation for the devices specified in the playbook.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            The function expects the following parameters in the configuration:
            - "ip_address_list": List of device IP addresses to be resynced.
            - "force_sync": (Optional) Whether to force sync the devices. Defaults to "False".
        """
    device_ips = self.get_device_ips_from_config_priority()
    input_device_ips = device_ips.copy()
    device_in_ccc = self.device_exists_in_ccc()
    for device_ip in input_device_ips:
        if device_ip not in device_in_ccc:
            input_device_ips.remove(device_ip)
    ap_devices = self.get_ap_devices(input_device_ips)
    self.log('AP Devices from the playbook input are: {0}'.format(str(ap_devices)), 'INFO')
    if ap_devices:
        for ap_ip in ap_devices:
            input_device_ips.remove(ap_ip)
        self.log("Following devices {0} are AP, so can't perform resync operation.".format(str(ap_devices)), 'WARNING')
    if not input_device_ips:
        self.msg = 'Cannot perform the Resync operation as the device(s) with IP(s) {0} are not present in Cisco Catalyst Center'.format(str(device_ips))
        self.status = 'success'
        self.result['changed'] = False
        self.result['response'] = self.msg
        self.log(self.msg, 'WARNING')
        return self
    device_ids = self.get_device_ids(input_device_ips)
    try:
        force_sync = self.config[0].get('force_sync', False)
        resync_param_dict = {'payload': device_ids, 'force_sync': force_sync}
        response = self.dnac._exec(family='devices', function='sync_devices_using_forcesync', op_modifies=True, params=resync_param_dict)
        self.log("Received API response from 'sync_devices_using_forcesync': {0}".format(str(response)), 'DEBUG')
        if response and isinstance(response, dict):
            task_id = response.get('response').get('taskId')
            while True:
                execution_details = self.get_task_details(task_id)
                if 'Synced' in execution_details.get('progress'):
                    self.status = 'success'
                    self.result['changed'] = True
                    self.result['response'] = execution_details
                    self.msg = 'Devices have been successfully resynced. Devices resynced: {0}'.format(str(input_device_ips))
                    self.log(self.msg, 'INFO')
                    break
                elif execution_details.get('isError'):
                    self.status = 'failed'
                    failure_reason = execution_details.get('failureReason')
                    if failure_reason:
                        self.msg = 'Device resynced get failed because of {0}'.format(failure_reason)
                    else:
                        self.msg = 'Device resynced get failed.'
                    self.log(self.msg, 'ERROR')
                    break
    except Exception as e:
        self.status = 'failed'
        error_message = 'Error while resyncing device in Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'ERROR')
    return self