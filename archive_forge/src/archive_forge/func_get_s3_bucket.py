from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_s3_bucket(self):
    api = 'protocols/s3/buckets'
    fields = 'name,svm.name,size,comment,volume.uuid,policy,policy.statements,qos_policy'
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 12, 1):
        fields += ',audit_event_selector,type,nas_path'
    elif self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
        fields += ',audit_event_selector'
    params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': fields}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return self.form_current(record) if record else None