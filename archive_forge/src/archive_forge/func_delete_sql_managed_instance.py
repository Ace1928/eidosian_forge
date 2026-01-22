from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def delete_sql_managed_instance(self):
    try:
        response = self.sql_client.managed_instances.begin_delete(self.resource_group, self.name)
    except Exception as exc:
        self.fail('Error when deleting SQL managed instance {0}: {1}'.format(self.name, exc))