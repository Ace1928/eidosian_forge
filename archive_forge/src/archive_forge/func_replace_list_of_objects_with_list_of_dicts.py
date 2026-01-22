from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def replace_list_of_objects_with_list_of_dicts(adict, key):
    if adict.get(key):
        adict[key] = [vars(x) for x in adict[key]]