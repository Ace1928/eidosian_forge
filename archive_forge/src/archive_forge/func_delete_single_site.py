from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def delete_single_site(self, site_id, site_name):
    """"
        Delete a single site in the Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            site_id (str): The ID of the site to be deleted.
            site_name (str): The name of the site to be deleted.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function initiates the deletion of a site in the Cisco Catalyst Center by calling the delete API.
            If the deletion is successful, the result is marked as changed, and the status is set to "success."
            If an error occurs during the deletion process, the status is set to "failed," and the log contains
            details about the error.
        """
    try:
        response = self.dnac._exec(family='sites', function='delete_site', params={'site_id': site_id})
        if response and isinstance(response, dict):
            self.log("Received API response from 'delete_site': {0}".format(str(response)), 'DEBUG')
            executionid = response.get('executionId')
            while True:
                execution_details = self.get_execution_details(executionid)
                if execution_details.get('status') == 'SUCCESS':
                    self.msg = "Site '{0}' deleted successfully".format(site_name)
                    self.result['changed'] = True
                    self.result['response'] = self.msg
                    self.status = 'success'
                    self.log(self.msg, 'INFO')
                    break
                elif execution_details.get('bapiError'):
                    self.log("Error response for 'delete_site' execution: {0}".format(execution_details.get('bapiError')), 'ERROR')
                    self.module.fail_json(msg=execution_details.get('bapiError'), response=execution_details)
                    break
    except Exception as e:
        self.status = 'failed'
        self.msg = "Exception occurred while deleting site '{0}' due to: {1}".format(site_name, str(e))
        self.log(self.msg, 'ERROR')
    return self