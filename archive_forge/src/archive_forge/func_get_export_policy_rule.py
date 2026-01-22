from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_export_policy_rule(self, rule_index):
    """
        Return details about the export policy rule
        If rule_index is None, fetch policy based on attributes
        :param:
            name : Name of the export_policy
        :return: Details about the export_policy. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_export_policy_rule_rest(rule_index)
    result = None
    rule_iter = netapp_utils.zapi.NaElement('export-rule-get-iter')
    query = self.set_query_parameters(rule_index)
    rule_iter.translate_struct(query)
    try:
        result = self.server.invoke_successfully(rule_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting export policy rule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if result is not None and result.get_child_by_name('num-records') and (int(result.get_child_content('num-records')) >= 1):
        if rule_index is None:
            return self.match_export_policy_rule_exactly(result.get_child_by_name('attributes-list').get_children(), query, is_rest=False)
        return self.zapi_export_rule_info_to_dict(result.get_child_by_name('attributes-list').get_child_by_name('export-rule-info'))
    return None