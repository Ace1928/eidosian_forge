from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
def create_update_applicationgateway(self):
    """
        Creates or updates Application Gateway with the specified configuration.

        :return: deserialized Application Gateway instance state dictionary
        """
    self.log('Creating / Updating the Application Gateway instance {0}'.format(self.name))
    try:
        response = self.network_client.application_gateways.begin_create_or_update(resource_group_name=self.resource_group, application_gateway_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Application Gateway instance.')
        self.fail('Error creating the Application Gateway instance: {0}'.format(str(exc)))
    return response.as_dict()