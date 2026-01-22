from __future__ import absolute_import, division, print_function
import time
def delete_mariadbdatabase(self):
    """
        Deletes specified MariaDB Database instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the MariaDB Database instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.databases.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the MariaDB Database instance.')
        self.fail('Error deleting the MariaDB Database instance: {0}'.format(str(e)))
    return True