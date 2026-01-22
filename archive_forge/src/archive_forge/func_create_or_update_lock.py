from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_lock(self, scope, lock):
    try:
        return self.lock_client.management_locks.create_or_update_by_scope(scope, self.name, lock)
    except Exception as exc:
        self.fail('Error when creating or updating lock {0} for {1}: {2}'.format(self.name, scope, exc.message))