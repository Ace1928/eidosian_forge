from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_applicationsecuritygroup(self):
    """
        Create or update Application Security Group.

        :return: deserialized Application Security Group instance state dictionary
        """
    self.log('Creating / Updating the Application Security Group instance {0}'.format(self.name))
    param = dict(name=self.name, tags=self.tags, location=self.location)
    try:
        response = self.network_client.application_security_groups.begin_create_or_update(resource_group_name=self.resource_group, application_security_group_name=self.name, parameters=param)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error creating/updating Application Security Group instance.')
        self.fail('Error creating/updating Application Security Group instance: {0}'.format(str(exc)))
    return response.as_dict()