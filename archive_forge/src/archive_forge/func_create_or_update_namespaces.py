from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import time
def create_or_update_namespaces(self):
    """
        create or update namespaces
        """
    try:
        namespace_params = EHNamespace(location=self.location, sku=Sku(name=self.sku), tags=self.tags)
        result = self.event_hub_client.namespaces.begin_create_or_update(self.resource_group, self.namespace_name, namespace_params)
        namespace = self.event_hub_client.namespaces.get(self.resource_group, self.namespace_name)
        while namespace.provisioning_state == 'Created':
            time.sleep(30)
            namespace = self.event_hub_client.namespaces.get(self.resource_group, self.namespace_name)
    except Exception as ex:
        self.fail('Failed to create namespace {0} in resource group {1}: {2}'.format(self.namespace_name, self.resource_group, str(ex)))
    return namespace_to_dict(namespace)