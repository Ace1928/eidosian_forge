from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_on_access_policy_rest(self):
    self.svm_uuid = self.get_svm_uuid()
    if self.svm_uuid is None:
        self.module.fail_json(msg='Error: vserver %s not found' % self.parameters['vserver'])
    api = 'protocols/vscan/%s/on-access-policies' % self.svm_uuid
    query = {'name': self.parameters['policy_name']}
    fields = 'svm,name,mandatory,scope,enabled'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error searching Vscan on Access Policy %s: %s' % (self.parameters['policy_name'], to_native(error)))
    if record:
        return {'max_file_size': self.na_helper.safe_get(record, ['scope', 'max_file_size']), 'vserver': self.na_helper.safe_get(record, ['svm', 'name']), 'policy_name': record['name'], 'is_scan_mandatory': record['mandatory'], 'policy_status': record['enabled'], 'scan_files_with_no_ext': self.na_helper.safe_get(record, ['scope', 'scan_without_extension']), 'file_ext_to_exclude': self.na_helper.safe_get(record, ['scope', 'exclude_extensions']), 'file_ext_to_include': self.na_helper.safe_get(record, ['scope', 'include_extensions']), 'paths_to_exclude': self.na_helper.safe_get(record, ['scope', 'exclude_paths']), 'scan_readonly_volumes': self.na_helper.safe_get(record, ['scope', 'scan_readonly_volumes']), 'only_execute_access': self.na_helper.safe_get(record, ['scope', 'only_execute_access'])}
    return None