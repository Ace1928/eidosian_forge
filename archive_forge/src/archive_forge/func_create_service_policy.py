from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_service_policy(self):
    api = 'network/ip/service-policies'
    body = {'name': self.parameters['name']}
    if self.parameters.get('vserver') is not None:
        body['svm.name'] = self.parameters['vserver']
    for attr in ('ipspace', 'scope', 'services'):
        value = self.parameters.get(attr)
        if value is not None:
            body[attr] = value
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        msg = 'Error in create_service_policy: %s' % error
        self.module.fail_json(msg=msg)