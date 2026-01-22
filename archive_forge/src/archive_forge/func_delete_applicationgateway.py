from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def delete_applicationgateway(self):
    """
        Deletes specified Application Gateway instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Application Gateway instance {0}'.format(self.name))
    try:
        response = self.network_client.application_gateways.begin_delete(resource_group_name=self.resource_group, application_gateway_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Application Gateway instance.')
        self.fail('Error deleting the Application Gateway instance: {0}'.format(str(e)))
    return True