from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_image_params(self, params):
    """
        Get image name and the confirmation whether it's tagged golden or not

        Parameters:
          - self: The instance of the class containing the 'config' attribute
                  to be validated.
          - params: The validated params passed from the playbook.
        Returns:
          The method returns an instance of the class with updated attributes:
          - image_params: A dictionary containing all the values indicating
                          name of the image and its golden image status.
        Example:
          Post creation of the validated input, it fetches the required
          paramters and stores it for further processing and calling the
          parameters in other APIs.
        """
    image_params = {'image_name': params.get('image_name'), 'is_tagged_golden': params.get('golden_image')}
    self.log('Image details are {0}'.format(str(image_params)), 'INFO')
    return image_params