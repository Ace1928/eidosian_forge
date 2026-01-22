from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_schedule(self):
    """
        Creates or updates Schedule with the specified configuration.

        :return: deserialized Schedule instance state dictionary
        """
    self.log('Creating / Updating the Schedule instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.schedules.create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, schedule=self.schedule)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Schedule instance.')
        self.fail('Error creating the Schedule instance: {0}'.format(str(exc)))
    return response.as_dict()