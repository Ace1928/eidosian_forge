from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_or_install_certificate(self, validate_only=False):
    """
        Create or install certificate
        :return: message (should be empty dict)
        """
    required_keys = ['type', 'common_name']
    if validate_only:
        if not set(required_keys).issubset(set(self.parameters.keys())):
            self.module.fail_json(msg='Error creating or installing certificate: one or more of the following options are missing: %s' % ', '.join(required_keys))
        return
    optional_keys = ['public_certificate', 'private_key', 'expiry_time', 'key_size', 'hash_function', 'intermediate_certificates']
    if not self.ignore_name_param:
        optional_keys.append('name')
    body = {}
    if self.parameters.get('svm') is not None:
        body['svm'] = {'name': self.parameters['svm']}
    for key in required_keys + optional_keys:
        if self.parameters.get(key) is not None:
            body[key] = self.parameters[key]
    params = {'return_records': 'true'}
    api = 'security/certificates'
    message, error = self.rest_api.post(api, body, params)
    if error:
        if self.parameters.get('svm') is None and error.get('target') == 'uuid':
            error['target'] = 'cluster'
        if error.get('message') == 'duplicate entry':
            error['message'] += '.  Same certificate may already exist under a different name.'
        self.module.fail_json(msg='Error creating or installing certificate: %s' % error)
    return message