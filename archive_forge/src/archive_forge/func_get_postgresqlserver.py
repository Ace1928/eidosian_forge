from __future__ import absolute_import, division, print_function
import time
def get_postgresqlserver(self):
    """
        Gets the properties of the specified PostgreSQL Server.

        :return: deserialized PostgreSQL Server instance state dictionary
        """
    self.log('Checking if the PostgreSQL Server instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.postgresql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('PostgreSQL Server instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the PostgreSQL Server instance.')
    if found is True:
        return response.as_dict()
    return False