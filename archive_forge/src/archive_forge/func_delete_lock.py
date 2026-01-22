from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_lock(self, scope):
    try:
        return self.lock_client.management_locks.delete_by_scope(scope, self.name)
    except Exception as exc:
        self.fail('Error when deleting lock {0} for {1}: {2}'.format(self.name, scope, exc.message))