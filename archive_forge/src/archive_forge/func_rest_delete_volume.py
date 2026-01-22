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
def rest_delete_volume(self, current):
    """
        Delete the volume using REST DELETE method (it scrubs better than ZAPI).
        """
    uuid = self.parameters['uuid']
    if uuid is None:
        self.module.fail_json(msg='Could not read UUID for volume %s in delete.' % self.parameters['name'])
    unmount_error = self.volume_unmount_rest(fail_on_error=False) if current.get('junction_path') else None
    dummy, error = rest_generic.delete_async(self.rest_api, 'storage/volumes', uuid, job_timeout=self.parameters['time_out'])
    self.na_helper.fail_on_error(error, previous_errors=['Error unmounting volume: %s' % unmount_error] if unmount_error else None)
    if unmount_error:
        self.module.warn('Volume was successfully deleted though unmount failed with: %s' % unmount_error)