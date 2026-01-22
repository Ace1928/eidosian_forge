from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
@staticmethod
def dict_from_volume_object(volume_object):

    def replace_list_of_objects_with_list_of_dicts(adict, key):
        if adict.get(key):
            adict[key] = [vars(x) for x in adict[key]]
    current_dict = vars(volume_object)
    attr = 'subnet_id'
    if attr in current_dict:
        current_dict['subnet_name'] = current_dict.pop(attr).split('/')[-1]
    attr = 'mount_targets'
    replace_list_of_objects_with_list_of_dicts(current_dict, attr)
    attr = 'export_policy'
    if current_dict.get(attr):
        attr_dict = vars(current_dict[attr])
        replace_list_of_objects_with_list_of_dicts(attr_dict, 'rules')
        current_dict[attr] = attr_dict
    return current_dict