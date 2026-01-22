from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_export_policy_rule_rest(self, params, rule_index, rename=False):
    api = 'protocols/nfs/export-policies/%s/rules' % self.policy_id
    query = {'new_index': self.parameters['rule_index']} if rename else None
    dummy, error = rest_generic.patch_async(self.rest_api, api, rule_index, self.create_body(params), query)
    if error:
        self.module.fail_json(msg='Error on modifying export policy Rule: %s' % error)