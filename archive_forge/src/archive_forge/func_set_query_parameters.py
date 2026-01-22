from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def set_query_parameters(self, rule_index):
    """
        Return dictionary of query parameters and
        :return:
        """
    query = {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver']}
    if rule_index is not None:
        query['rule-index'] = rule_index
    else:
        for item_key, value in self.parameters.items():
            zapi_key = None
            if item_key in self.na_helper.zapi_string_keys and item_key != 'client_match':
                zapi_key = self.na_helper.zapi_string_keys[item_key]
            elif item_key in self.na_helper.zapi_bool_keys:
                zapi_key = self.na_helper.zapi_bool_keys[item_key]
                value = self.na_helper.get_value_for_bool(from_zapi=False, value=value)
            elif item_key in self.na_helper.zapi_list_keys:
                zapi_key, child_key = self.na_helper.zapi_list_keys[item_key]
                value = [{child_key: item} for item in value] if value else None
            if zapi_key:
                self.set_dict_when_not_none(query, zapi_key, value)
    return {'query': {'export-rule-info': query}}