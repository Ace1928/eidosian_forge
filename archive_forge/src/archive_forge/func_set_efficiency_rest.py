from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def set_efficiency_rest(self):
    body = {}
    if self.parameters.get('efficiency_policy') is not None:
        body['efficiency.policy.name'] = self.parameters['efficiency_policy']
    if self.get_compression():
        body['efficiency.compression'] = self.get_compression()
    if not body:
        return
    dummy, error = self.volume_rest_patch(body)
    if error:
        self.module.fail_json(msg='Error setting efficiency for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())