from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def delete_sqldatabase(self):
    """
        Deletes specified SQL Database instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the SQL Database instance {0}'.format(self.name))
    try:
        response = self.sql_client.databases.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the SQL Database instance.')
        self.fail('Error deleting the SQL Database instance: {0}'.format(str(e)))
    return True