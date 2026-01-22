from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def verify_diff_activated(self):
    """
        Verify the activation status of a software image in Cisco Catalyst Center.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This method verifies the activation status of a software image in Cisco Catalyst Center and retrieves the image ID and name from
            the input. If activation device ID is provided, it checks the activation status for that specific device. Based on activation status
            a corresponding message is logged.
        """
    image_id = self.have.get('activation_image_id')
    image_name = self.get_image_name_from_id(image_id)
    if self.have.get('activation_device_id'):
        if self.single_device_activation:
            self.msg = "The requested image '{0}', associated with the device ID '{1}', has been successfully activated in the Cisco Catalyst\n                         Center and its status has been verified.".format(image_name, self.have.get('activation_device_id'))
            self.log(self.msg, 'INFO')
        else:
            self.log("Mismatch between the playbook's input for activating the image '{0}' on the device with ID '{1}' and the actual state in\n                         the Cisco Catalyst Center suggests that the activation task might not have been executed\n                         successfully.".format(image_name, self.have.get('activation_device_id')), 'INFO')
    elif self.complete_successful_activation:
        self.msg = "The requested image '{0}', with ID '{1}', has been successfully activated on all devices within the specified site in the\n                     Cisco Catalyst Center.".format(image_name, image_id)
        self.log(self.msg, 'INFO')
    elif self.partial_successful_activation:
        self.msg = '"The requested image \'{0}\', with ID \'{1}\', has been partially activated on some devices in the Cisco\n                     Catalyst Center.'.format(image_name, image_id)
        self.log(self.msg, 'INFO')
    else:
        self.msg = "The activation of the requested image '{0}', with ID '{1}', failed on devices in the Cisco\n                     Catalyst Center.".format(image_name, image_id)
        self.log(self.msg, 'INFO')
    return self