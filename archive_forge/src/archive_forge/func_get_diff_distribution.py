from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_diff_distribution(self):
    """
        Get image distribution parameters from the playbook and trigger image distribution.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function retrieves image distribution parameters from the playbook's 'distribution_details' and triggers
            the distribution of the specified software image to the specified device. It monitors the distribution task's
            progress and updates the 'result' dictionary. If the operation is successful, 'changed' is set to True.
        """
    distribution_details = self.want.get('distribution_details')
    site_name = distribution_details.get('site_name')
    device_family = distribution_details.get('device_family_name')
    device_role = distribution_details.get('device_role', 'ALL')
    device_series_name = distribution_details.get('device_series_name')
    device_uuid_list = self.get_device_uuids(site_name, device_family, device_role, device_series_name)
    image_id = self.have.get('distribution_image_id')
    self.complete_successful_distribution = False
    self.partial_successful_distribution = False
    self.single_device_distribution = False
    if self.have.get('distribution_device_id'):
        distribution_params = dict(payload=[dict(deviceUuid=self.have.get('distribution_device_id'), imageUuid=image_id)])
        self.log('Distribution Params: {0}'.format(str(distribution_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='trigger_software_image_distribution', op_modifies=True, params=distribution_params)
        self.log("Received API response from 'trigger_software_image_distribution': {0}".format(str(response)), 'DEBUG')
        if response:
            task_details = {}
            task_id = response.get('response').get('taskId')
            while True:
                task_details = self.get_task_details(task_id)
                if not task_details.get('isError') and 'completed successfully' in task_details.get('progress'):
                    self.result['changed'] = True
                    self.status = 'success'
                    self.single_device_distribution = True
                    self.result['msg'] = 'Image with Id {0} Distributed Successfully'.format(image_id)
                    break
                if task_details.get('isError'):
                    self.status = 'failed'
                    self.msg = 'Image with Id {0} Distribution Failed'.format(image_id)
                    self.log(self.msg, 'ERROR')
                    self.result['response'] = task_details
                    break
                self.result['response'] = task_details if task_details else response
        return self
    if len(device_uuid_list) == 0:
        self.status = 'success'
        self.msg = 'The SWIM image distribution task could not proceed because no eligible devices were found.'
        self.result['msg'] = self.msg
        self.log(self.msg, 'WARNING')
        return self
    self.log('Device UUIDs involved in Image Distribution: {0}'.format(str(device_uuid_list)), 'INFO')
    distribution_task_dict = {}
    for device_uuid in device_uuid_list:
        device_management_ip = self.get_device_ip_from_id(device_uuid)
        distribution_params = dict(payload=[dict(deviceUuid=device_uuid, imageUuid=image_id)])
        self.log('Distribution Params: {0}'.format(str(distribution_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='trigger_software_image_distribution', op_modifies=True, params=distribution_params)
        self.log("Received API response from 'trigger_software_image_distribution': {0}".format(str(response)), 'DEBUG')
        if response:
            task_details = {}
            task_id = response.get('response').get('taskId')
            distribution_task_dict[device_management_ip] = task_id
    device_ips_list, device_distribution_count = self.check_swim_task_status(distribution_task_dict, 'Distribution')
    if device_distribution_count == 0:
        self.status = 'failed'
        self.msg = 'Image with Id {0} Distribution Failed for all devices'.format(image_id)
    elif device_distribution_count == len(device_uuid_list):
        self.result['changed'] = True
        self.status = 'success'
        self.complete_successful_distribution = True
        self.msg = 'Image with Id {0} Distributed Successfully for all devices'.format(image_id)
    else:
        self.result['changed'] = True
        self.status = 'success'
        self.partial_successful_distribution = False
        self.msg = "Image with Id '{0}' Distributed and partially successfull".format(image_id)
        self.log('For device(s) {0} image Distribution gets failed'.format(str(device_ips_list)), 'CRITICAL')
    self.result['msg'] = self.msg
    self.log(self.msg, 'INFO')
    return self