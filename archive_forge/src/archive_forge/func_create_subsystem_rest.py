from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_subsystem_rest(self):
    api = 'protocols/nvme/subsystems'
    body = {'svm.name': self.parameters['vserver'], 'os_type': self.parameters['ostype'], 'name': self.parameters['subsystem']}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating subsystem for vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())