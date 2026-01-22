from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_applicationsecuritygroup(self):
    """
        Deletes specified Application Security Group instance.

        :return: True
        """
    self.log('Deleting the Application Security Group instance {0}'.format(self.name))
    try:
        response = self.network_client.application_security_groups.begin_delete(resource_group_name=self.resource_group, application_security_group_name=self.name)
    except Exception as e:
        self.log('Error deleting the Application Security Group instance.')
        self.fail('Error deleting the Application Security Group instance: {0}'.format(str(e)))
    return True