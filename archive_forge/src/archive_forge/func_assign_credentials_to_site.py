from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def assign_credentials_to_site(self):
    """
        Assign Global Device Credential to the Cisco DNA
        Center based on the provided playbook details.
        Check the return value of the API with check_return_status().

        Parameters:
            self

        Returns:
            self
        """
    result_assign_credential = self.result.get('response')[0].get('assignCredential')
    credential_params = self.want.get('assign_credentials')
    final_response = []
    self.log('Assigning device credential to site API input parameters: {0}'.format(credential_params), 'DEBUG')
    if not credential_params:
        result_assign_credential.update({'No Assign Credentials': {'response': 'No Response', 'msg': 'No Assignment is available'}})
        self.msg = 'No Assignment is available'
        self.status = 'success'
        return self
    site_ids = self.want.get('site_id')
    for site_id in site_ids:
        credential_params.update({'site_id': site_id})
        final_response.append(copy.deepcopy(credential_params))
        response = self.dnac._exec(family='network_settings', function='assign_device_credential_to_site_v2', params=credential_params)
        self.log("Received API response for 'assign_device_credential_to_site_v2': {0}".format(response), 'DEBUG')
        validation_string = 'desired common settings operation successful'
        self.check_task_response_status(response, validation_string).check_return_status()
    self.log('Device credential assigned to site {0} is successfully.'.format(site_ids), 'INFO')
    self.log('Desired State for assign credentials to a site: {0}'.format(final_response), 'DEBUG')
    result_assign_credential.update({'Assign Credentials': {'response': final_response, 'msg': 'Device Credential Assigned to a site is Successfully'}})
    self.msg = 'Global Credential is assigned Successfully'
    self.status = 'success'
    return self