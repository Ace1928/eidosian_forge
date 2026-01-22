from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def zapi_export_rule_info_to_dict(self, rule_info):
    current = {}
    for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
        current[item_key] = rule_info.get_child_content(zapi_key)
        if item_key == 'client_match' and current[item_key]:
            current[item_key] = current[item_key].split(',')
    for item_key, zapi_key in self.na_helper.zapi_bool_keys.items():
        current[item_key] = self.na_helper.get_value_for_bool(from_zapi=True, value=rule_info[zapi_key])
    for item_key, zapi_key in self.na_helper.zapi_int_keys.items():
        current[item_key] = self.na_helper.get_value_for_int(from_zapi=True, value=rule_info[zapi_key])
    for item_key, zapi_key in self.na_helper.zapi_list_keys.items():
        parent, dummy = zapi_key
        current[item_key] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=rule_info.get_child_by_name(parent))
    return current