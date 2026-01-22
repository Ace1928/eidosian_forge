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
def create_volume_rest(self):
    body = self.create_volume_body_rest()
    dummy, error = rest_generic.post_async(self.rest_api, 'storage/volumes', body, job_timeout=self.parameters['time_out'])
    if error:
        self.module.fail_json(msg='Error creating volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if self.parameters.get('wait_for_completion'):
        self.wait_for_volume_online(sleep_time=5)