from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_export_policy_rule_rest(self):
    api = 'protocols/nfs/export-policies/%s/rules?return_records=true' % self.policy_id
    response, error = rest_generic.post_async(self.rest_api, api, self.create_body(self.parameters))
    if error:
        self.module.fail_json(msg='Error on creating export policy rule: %s' % error)
    rule_index = None
    if response and response.get('num_records') == 1:
        rule_index = self.na_helper.safe_get(response, ['records', 0, 'index'])
    if rule_index is None:
        self.module.fail_json(msg='Error on creating export policy rule, returned response is invalid: %s' % response)
    if self.parameters.get('rule_index'):
        self.modify_export_policy_rule_rest({}, rule_index, True)