from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_s3_service(self, extra_field=False):
    api = 'protocols/s3/services'
    fields = ','.join(('name', 'enabled', 'svm.uuid', 'comment', 'certificate.name'))
    if extra_field:
        fields += ',users'
    params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': fields}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if record:
        if self.na_helper.safe_get(record, ['certificate', 'name']):
            record['certificate_name'] = self.na_helper.safe_get(record, ['certificate', 'name'])
        return self.set_uuids(record)
    return None