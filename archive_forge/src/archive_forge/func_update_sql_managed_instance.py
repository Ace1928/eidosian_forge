from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def update_sql_managed_instance(self, parameters):
    try:
        response = self.sql_client.managed_instances.begin_update(resource_group_name=self.resource_group, managed_instance_name=self.name, parameters=parameters)
        try:
            response = self.sql_client.managed_instances.get(resource_group_name=self.resource_group, managed_instance_name=self.name)
        except ResourceNotFoundError:
            self.fail("The resource created failed, can't get the facts")
        return self.to_dict(response)
    except Exception as exc:
        self.fail('Error when updating SQL managed instance {0}: {1}'.format(self.name, exc.message))