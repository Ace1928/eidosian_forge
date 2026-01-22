from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def verify_diff_tagged(self):
    """
        Verify the Golden tagging status of a software image in Cisco Catalyst Center.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This method verifies the tagging status of a software image in Cisco Catalyst Center.
            It retrieves tagging details from the input, including the desired tagging status and image ID.
            Using the provided image ID, it obtains image parameters required for checking the image status.
            The method then queries Catalyst Center to get the golden tag status of the image.
            If the image status matches the desired tagging status, a success message is logged.
            If there is a mismatch between the playbook input and the Catalyst Center, a warning message is logged.
        """
    tagging_details = self.want.get('tagging_details')
    tag_image_golden = tagging_details.get('tagging')
    image_id = self.have.get('tagging_image_id')
    image_name = self.get_image_name_from_id(image_id)
    image_params = dict(image_id=self.have.get('tagging_image_id'), site_id=self.have.get('site_id'), device_family_identifier=self.have.get('device_family_identifier'), device_role=tagging_details.get('device_role', 'ALL').upper())
    self.log('Parameters for checking the status of image: {0}'.format(str(image_params)), 'INFO')
    response = self.dnac._exec(family='software_image_management_swim', function='get_golden_tag_status_of_an_image', op_modifies=True, params=image_params)
    self.log("Received API response from 'get_golden_tag_status_of_an_image': {0}".format(str(response)), 'DEBUG')
    response = response.get('response')
    if response:
        image_status = response['taggedGolden']
        if image_status == tag_image_golden:
            if tag_image_golden:
                self.msg = "The requested image '{0}' has been tagged as golden in the Cisco Catalyst Center and\n                             its status has been successfully verified.".format(image_name)
                self.log(self.msg, 'INFO')
            else:
                self.msg = "The requested image '{0}' has been un-tagged as golden in the Cisco Catalyst Center and\n                            image status has been verified.".format(image_name)
                self.log(self.msg, 'INFO')
    else:
        self.log('Mismatch between the playbook input for tagging/un-tagging image as golden and the Cisco Catalyst Center indicates that\n                        the tagging/un-tagging task was not executed successfully.', 'INFO')
    return self