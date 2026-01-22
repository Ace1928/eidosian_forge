from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import time
import json
def delete_vm_backup_policy(self):
    try:
        response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
    except Exception as e:
        self.log('Error attempting to delete Azure Backup policy.')
        self.fail('Error attempting to delete Azure Backup policy: {0}'.format(str(e)))