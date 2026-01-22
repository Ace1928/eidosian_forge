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
def rename_volume_rest(self):
    current = self.get_volume_rest(self.parameters['from_name'])
    body = {'name': self.parameters['name']}
    dummy, error = self.volume_rest_patch(body, uuid=current['uuid'])
    if error:
        self.module.fail_json(msg='Error changing name of volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())