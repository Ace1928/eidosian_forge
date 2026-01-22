from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def check_swim_task_status(self, swim_task_dict, swim_task_name):
    """
        Check the status of the SWIM (Software Image Management) task for each device.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            swim_task_dict (dict): A dictionary containing the mapping of device IP address to the respective task ID.
            swim_task_name (str): The name of the SWIM task being checked which is either Distribution or Activation.
        Returns:
            tuple: A tuple containing two elements:
                - device_ips_list (list): A list of device IP addresses for which the SWIM task failed.
                - device_count (int): The count of devices for which the SWIM task was successful.
        Description:
            This function iterates through the distribution_task_dict, which contains the mapping of
            device IP address to their respective task ID. It checks the status of the SWIM task for each device by
            repeatedly querying for task details until the task is either completed successfully or fails. If the task
            is successful, the device count is incremented. If the task fails, an error message is logged, and the device
            IP is appended to the device_ips_list and return a tuple containing the device_ips_list and device_count.
        """
    device_ips_list = []
    device_count = 0
    for device_ip, task_id in swim_task_dict.items():
        start_time = time.time()
        max_timeout = self.params.get('dnac_api_task_timeout')
        while True:
            end_time = time.time()
            if end_time - start_time >= max_timeout:
                self.log("Max timeout of {0} sec has reached for the task id '{1}' for the device '{2}' and unexpected\n                                 task status so moving out to next task id".format(max_timeout, task_id, device_ip), 'WARNING')
                device_ips_list.append(device_ip)
                break
            task_details = self.get_task_details(task_id)
            if not task_details.get('isError') and 'completed successfully' in task_details.get('progress'):
                self.result['changed'] = True
                self.status = 'success'
                self.log("Image {0} successfully for the device '{1}".format(swim_task_name, device_ip), 'INFO')
                device_count += 1
                break
            if task_details.get('isError'):
                error_msg = "Image {0} gets failed for the device '{1}'".format(swim_task_name, device_ip)
                self.log(error_msg, 'ERROR')
                self.result['response'] = task_details
                device_ips_list.append(device_ip)
                break
            time.sleep(self.params.get('dnac_task_poll_interval'))
    return (device_ips_list, device_count)