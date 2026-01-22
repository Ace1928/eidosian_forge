from __future__ import absolute_import, division, print_function
import time
def delete_mysqlserver(self):
    """
        Deletes specified MySQL Server instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the MySQL Server instance {0}'.format(self.name))
    try:
        response = self.mysql_client.servers.begin_delete(resource_group_name=self.resource_group, server_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the MySQL Server instance.')
        self.fail('Error deleting the MySQL Server instance: {0}'.format(str(e)))
    return True