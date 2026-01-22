from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_ipsec_ca_certificate(self):
    """Create IPsec CA certifcate"""
    api = 'security/ipsec/ca-certificates'
    body = {'certificate.uuid': self.uuid}
    if self.parameters.get('svm'):
        body['svm.name'] = self.parameters['svm']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error adding security IPsec CA certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())