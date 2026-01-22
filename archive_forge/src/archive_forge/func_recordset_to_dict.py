from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, HAS_AZURE
def recordset_to_dict(self, recordset):
    result = recordset.as_dict()
    result['type'] = result['type'].strip('Microsoft.Network/dnszones/')
    return result