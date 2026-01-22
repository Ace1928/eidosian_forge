from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def firmware_image_get(self, node_name):
    """
        Get current firmware image info
        :return: True if query successful, else return None
        """
    firmware_image_get_iter = self.firmware_image_get_iter()
    try:
        result = self.server.invoke_successfully(firmware_image_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching firmware image details: %s: %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        sp_info = result.get_child_by_name('attributes-list').get_child_by_name('service-processor-info')
        return sp_info.get_child_content('firmware-version')
    return None