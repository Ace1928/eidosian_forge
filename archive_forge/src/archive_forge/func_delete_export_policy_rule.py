from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def delete_export_policy_rule(self, rule_index):
    """
        delete rule for the export policy.
        """
    if self.use_rest:
        return self.delete_export_policy_rule_rest(rule_index)
    export_rule_delete = netapp_utils.zapi.NaElement.create_node_with_children('export-rule-destroy', **{'policy-name': self.parameters['name'], 'rule-index': str(rule_index)})
    try:
        self.server.invoke_successfully(export_rule_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting export policy rule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())