from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def delete_storage_account(self, resource_group, name):
    self.log('Delete storage account {0}'.format(name))
    self.results['actions'].append('Deleted storage account {0}'.format(name))
    try:
        self.storage_client.storage_accounts.delete(self.resource_group, name)
    except Exception as exc:
        self.fail('Error deleting storage account {0} - {1}'.format(name, str(exc)))
    return True