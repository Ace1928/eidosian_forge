from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_elastic_pool(self, parameters):
    """
        Creates or updates SQL Elastic Pool with the specified configuration.

        :return: deserialized SQL Elastic Pool instance state dictionary
        """
    self.log('Creating / Updating the SQL Elastic Pool instance {0}'.format(self.name))
    try:
        response = self.sql_client.elastic_pools.begin_update(resource_group_name=self.resource_group, server_name=self.server_name, elastic_pool_name=self.name, parameters=parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the SQL Elastic Pool instance.')
        self.fail('Error creating the SQL Elastic Pool instance: {0}'.format(str(exc)))
    return self.format_item(response)