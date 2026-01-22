from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_diff_import(self):
    """
        Check the image import type and fetch the image ID for the imported image for further use.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function checks the type of image import (URL or local) and proceeds with the import operation accordingly.
            It then monitors the import task's progress and updates the 'result' dictionary. If the operation is successful,
            'changed' is set to True.
            Additionally, if tagging, distribution, or activation details are provided, it fetches the image ID for the
            imported image and stores it in the 'have' dictionary for later use.
        """
    try:
        import_type = self.want.get('import_type')
        if not import_type:
            self.status = 'success'
            self.msg = 'Error: Details required for importing SWIM image. Please provide the necessary information.'
            self.result['msg'] = self.msg
            self.log(self.msg, 'WARNING')
            self.result['changed'] = False
            return self
        if import_type == 'remote':
            image_name = self.want.get('url_import_details').get('payload')[0].get('source_url')
        else:
            image_name = self.want.get('local_import_details').get('file_path')
        name = image_name.split('/')[-1]
        image_exist = self.is_image_exist(name)
        import_key_mapping = {'source_url': 'sourceURL', 'image_family': 'imageFamily', 'application_type': 'applicationType', 'is_third_party': 'thirdParty'}
        if image_exist:
            image_id = self.get_image_id(name)
            self.have['imported_image_id'] = image_id
            self.msg = "Image '{0}' already exists in the Cisco Catalyst Center".format(name)
            self.result['msg'] = self.msg
            self.log(self.msg, 'INFO')
            self.status = 'success'
            self.result['changed'] = False
            return self
        if self.want.get('import_type') == 'remote':
            import_payload_dict = {}
            temp_payload = self.want.get('url_import_details').get('payload')[0]
            keys_to_change = list(import_key_mapping.keys())
            for key, val in temp_payload.items():
                if key in keys_to_change:
                    api_key_name = import_key_mapping[key]
                    import_payload_dict[api_key_name] = val
            import_image_payload = [import_payload_dict]
            import_params = dict(payload=import_image_payload, scheduleAt=self.want.get('url_import_details').get('schedule_at'), scheduleDesc=self.want.get('url_import_details').get('schedule_desc'), scheduleOrigin=self.want.get('url_import_details').get('schedule_origin'))
            import_function = 'import_software_image_via_url'
        else:
            file_path = self.want.get('local_import_details').get('file_path')
            import_params = dict(is_third_party=self.want.get('local_import_details').get('is_third_party'), third_party_vendor=self.want.get('local_import_details').get('third_party_vendor'), third_party_image_family=self.want.get('local_import_details').get('third_party_image_family'), third_party_application_type=self.want.get('local_import_details').get('third_party_application_type'), multipart_fields={'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/octet-stream')}, multipart_monitor_callback=None)
            import_function = 'import_local_software_image'
        response = self.dnac._exec(family='software_image_management_swim', function=import_function, op_modifies=True, params=import_params)
        self.log('Received API response from {0}: {1}'.format(import_function, str(response)), 'DEBUG')
        task_details = {}
        task_id = response.get('response').get('taskId')
        while True:
            task_details = self.get_task_details(task_id)
            name = image_name.split('/')[-1]
            if task_details and 'completed successfully' in task_details.get('progress').lower():
                self.result['changed'] = True
                self.status = 'success'
                self.msg = 'Swim Image {0} imported successfully'.format(name)
                self.result['msg'] = self.msg
                self.log(self.msg, 'INFO')
                break
            if task_details and task_details.get('isError'):
                if 'already exists' in task_details.get('failureReason', ''):
                    self.msg = 'SWIM Image {0} already exists in the Cisco Catalyst Center'.format(name)
                    self.result['msg'] = self.msg
                    self.log(self.msg, 'INFO')
                    self.status = 'success'
                    self.result['changed'] = False
                    break
                else:
                    self.status = 'failed'
                    self.msg = task_details.get('failureReason', 'SWIM Image {0} seems to be invalid'.format(image_name))
                    self.log(self.msg, 'WARNING')
                    self.result['response'] = self.msg
                    return self
        self.result['response'] = task_details if task_details else response
        image_name = image_name.split('/')[-1]
        image_id = self.get_image_id(image_name)
        self.have['imported_image_id'] = image_id
        return self
    except Exception as e:
        self.status = 'failed'
        self.msg = 'Error: Import image details are not provided in the playbook, or the Import Image API was not\n                 triggered successfully. Please ensure the necessary details are provided and verify the status of the Import Image process.'
        self.log(self.msg, 'ERROR')
        self.result['response'] = self.msg
    return self