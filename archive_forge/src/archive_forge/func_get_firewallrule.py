from __future__ import absolute_import, division, print_function
import time
def get_firewallrule(self):
    """
        Gets the properties of the specified MariaDB firewall rule.

        :return: deserialized MariaDB firewall rule instance state dictionary
        """
    self.log('Checking if the MariaDB firewall rule instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.mariadb_client.firewall_rules.get(resource_group_name=self.resource_group, server_name=self.server_name, firewall_rule_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('MariaDB firewall rule instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the MariaDB firewall rule instance.')
    if found is True:
        return response.as_dict()
    return False