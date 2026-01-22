from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_iscsi_service_rest(self):
    api = 'protocols/san/iscsi/services'
    body = {'svm.name': self.parameters['vserver'], 'enabled': True if self.parameters.get('service_state', 'started') == 'started' else False}
    if 'target_alias' in self.parameters:
        body['target.alias'] = self.parameters['target_alias']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating iscsi service: % s' % error)