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
def wait_for_volume_encryption_conversion(self):
    if self.use_rest:
        return self.wait_for_volume_encryption_conversion_rest()
    volume_encryption_conversion_iter = netapp_utils.zapi.NaElement('volume-encryption-conversion-get-iter')
    volume_encryption_conversion_info = netapp_utils.zapi.NaElement('volume-encryption-conversion-info')
    volume_encryption_conversion_info.add_new_child('volume', self.parameters['name'])
    volume_encryption_conversion_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(volume_encryption_conversion_info)
    volume_encryption_conversion_iter.add_child_elem(query)
    error = self.wait_for_task_completion(volume_encryption_conversion_iter, self.check_volume_encryption_conversion_state)
    if error:
        self.module.fail_json(msg='Error getting volume encryption_conversion status: %s' % to_native(error), exception=traceback.format_exc())