from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_export_policy_rule_rest(self, rule_index):
    self.set_export_policy_id_rest()
    if not self.policy_id:
        return None
    query = {'fields': 'anonymous_user,clients,index,protocols,ro_rule,rw_rule,superuser'}
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
        query['fields'] += ',ntfs_unix_security,allow_suid,chown_mode,allow_device_creation'
    if rule_index is None:
        return self.get_export_policy_rule_exact_match(query)
    api = 'protocols/nfs/export-policies/%s/rules/%s' % (self.policy_id, rule_index)
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        if "entry doesn't exist" in error:
            return None
        self.module.fail_json(msg='Error on fetching export policy rule: %s' % error)
    return self.filter_get_results(record) if record else None