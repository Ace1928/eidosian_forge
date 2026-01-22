from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_cli_params(self, cliDetails):
    """
        Format the CLI parameters for the CLI credential configuration in Cisco DNA Center.

        Parameters:
            cliDetails (list of dict) - Cisco DNA Center details containing CLI Credentials.

        Returns:
            cliCredential (list of dict) - Processed CLI credential data
            in the format suitable for the Cisco DNA Center config.
        """
    cliCredential = []
    for item in cliDetails:
        if item is None:
            cliCredential.append(None)
        else:
            value = {'username': item.get('username'), 'description': item.get('description'), 'id': item.get('id')}
            cliCredential.append(value)
    return cliCredential