from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_nvme_rest(self):
    api = 'protocols/nvme/services'
    body = {'svm.name': self.parameters['vserver']}
    if self.parameters.get('status_admin') is not None:
        body['enabled'] = self.parameters['status_admin']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating nvme for vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())