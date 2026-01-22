from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_reset_params(self):
    """
        Get the paramters needed for resetting the device in an errored state.
        Parameters:
          - self: The instance of the class containing the 'config'
                  attribute to be validated.
        Returns:
          The method returns an instance of the class with updated attributes:
          - reset_params: A dictionary needed for calling the PUT call
                          for update device details API.
        Example:
          The stored dictionary can be used to call the API update device details
        """
    reset_params = {'deviceResetList': [{'configList': [{'configId': self.have.get('template_id'), 'configParameters': [{'key': '', 'value': ''}]}], 'deviceId': self.have.get('device_id'), 'licenseLevel': '', 'licenseType': '', 'topOfStackSerialNumber': ''}]}
    self.log('Paramters used for resetting from errored state:{0}'.format(str(reset_params)), 'INFO')
    return reset_params