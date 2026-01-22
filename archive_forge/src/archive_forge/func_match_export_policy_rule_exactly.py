from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def match_export_policy_rule_exactly(self, records, query, is_rest):
    if not records:
        return None
    founds = []
    for record in records:
        record = self.filter_get_results(record) if is_rest else self.zapi_export_rule_info_to_dict(record)
        modify = self.na_helper.get_modified_attributes(record, self.parameters)
        modify.pop('rule_index', None)
        if not modify:
            founds.append(record)
    if founds and len(founds) > 1 and (not (self.parameters['state'] == 'absent' and self.parameters['force_delete_on_first_match'])):
        self.module.fail_json(msg='Error multiple records exist for query: %s.  Specify index to modify or delete a rule.  Found: %s' % (query, founds))
    return founds[0] if founds else None