from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def remove_autocreated_resources(self, tags):
    if tags:
        sa_name = tags.get('_own_sa_')
        nic_name = tags.get('_own_nic_')
        pip_name = tags.get('_own_pip_')
        nsg_name = tags.get('_own_nsg_')
        if sa_name:
            self.delete_storage_account(self.resource_group, sa_name)
        if nic_name:
            self.delete_nic(self.resource_group, nic_name)
        if pip_name:
            self.delete_pip(self.resource_group, pip_name)
        if nsg_name:
            self.delete_nsg(self.resource_group, nsg_name)