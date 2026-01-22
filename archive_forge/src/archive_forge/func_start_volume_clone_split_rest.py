from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def start_volume_clone_split_rest(self):
    if self.uuid is None:
        self.module.fail_json(msg='Error starting volume clone split %s: %s' % (self.parameters['name'], 'clone UUID is not set'))
    api = 'storage/volumes'
    body = {'clone.split_initiated': True}
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error starting volume clone split %s: %s' % (self.parameters['name'], to_native(error)))