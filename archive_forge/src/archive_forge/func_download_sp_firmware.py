from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def download_sp_firmware(self):
    if self.parameters.get('reboot_sp'):
        self.reboot_sp()
    if self.use_rest:
        return self.download_software_rest()
    self.download_sp_image()
    progress = self.download_sp_image_progress()
    if progress['phase'] == 'Download':
        while progress['run_status'] is not None and progress['run_status'] != 'Exited':
            time.sleep(10)
            progress = self.download_sp_image_progress()
        if progress['exit_status'] != 'Success':
            self.module.fail_json(msg=progress['exit_message'], exception=traceback.format_exc())
        return MSGS['dl_completed']
    return MSGS['no_action']