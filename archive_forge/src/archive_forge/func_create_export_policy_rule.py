from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_export_policy_rule(self):
    """
        create rule for the export policy.
        """
    if self.use_rest:
        return self.create_export_policy_rule_rest()
    export_rule_create = netapp_utils.zapi.NaElement('export-rule-create')
    self.add_parameters_for_create_or_modify(export_rule_create, self.parameters)
    try:
        self.server.invoke_successfully(export_rule_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating export policy rule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())