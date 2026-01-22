from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def trigger_export_api(self, payload_params):
    """
        Triggers the export API to generate a CSV file containing device details based on the given payload parameters.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            payload_params (dict): A dictionary containing parameters required for the export API.
        Returns:
            dict: The response from the export API, including information about the task and file ID.
                If the export is successful, the CSV file can be downloaded using the file ID.
        Description:
            The function initiates the export API in Cisco Catalyst Center to generate a CSV file containing detailed information
            about devices.The response from the API includes task details and a file ID.
        """
    response = self.dnac._exec(family='devices', function='export_device_list', op_modifies=True, params=payload_params)
    self.log("Received API response from 'export_device_list': {0}".format(str(response)), 'DEBUG')
    response = response.get('response')
    task_id = response.get('taskId')
    while True:
        execution_details = self.get_task_details(task_id)
        if execution_details.get('additionalStatusURL'):
            file_id = execution_details.get('additionalStatusURL').split('/')[-1]
            break
        elif execution_details.get('isError'):
            self.status = 'failed'
            failure_reason = execution_details.get('failureReason')
            if failure_reason:
                self.msg = "Could not get the File ID because of {0} so can't export device details in csv file".format(failure_reason)
            else:
                self.msg = "Could not get the File ID so can't export device details in csv file"
            self.log(self.msg, 'ERROR')
            return response
    response = self.dnac._exec(family='file', function='download_a_file_by_fileid', op_modifies=True, params={'file_id': file_id})
    self.log("Received API response from 'download_a_file_by_fileid': {0}".format(str(response)), 'DEBUG')
    return response