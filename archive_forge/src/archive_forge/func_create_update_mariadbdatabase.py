from __future__ import absolute_import, division, print_function
import time
def create_update_mariadbdatabase(self):
    """
        Creates or updates MariaDB Database with the specified configuration.

        :return: deserialized MariaDB Database instance state dictionary
        """
    self.log('Creating / Updating the MariaDB Database instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.databases.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the MariaDB Database instance.')
        self.fail('Error creating the MariaDB Database instance: {0}'.format(str(exc)))
    return response.as_dict()