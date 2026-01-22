from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_s3_service(self):
    api = 'protocols/s3/services'
    body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['name']}
    if self.parameters.get('enabled') is not None:
        body['enabled'] = self.parameters['enabled']
    if self.parameters.get('comment'):
        body['comment'] = self.parameters['comment']
    if self.parameters.get('certificate_name'):
        body['certificate.name'] = self.parameters['certificate_name']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())