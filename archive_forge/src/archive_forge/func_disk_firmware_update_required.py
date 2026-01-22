from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def disk_firmware_update_required(self):
    """
        Check weather disk firmware upgrade is required or not
        :return: True if the firmware upgrade is required
        """
    disk_firmware_info = self.disk_firmware_info_get()
    return any((disk_firmware_info[disk] != self.parameters['disk_fw'] for disk in disk_firmware_info))