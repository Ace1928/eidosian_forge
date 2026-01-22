from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def update_software(self):
    if self.use_rest:
        return self.update_software_rest()
    self.cluster_image_update()
    cluster_update_progress = {}
    time_left = self.parameters['timeout']
    polling_interval = 25
    while time_left > 0 and cluster_update_progress.get('overall_status', 'in_progress') == 'in_progress':
        time.sleep(polling_interval)
        time_left -= polling_interval
        cluster_update_progress = self.cluster_image_update_progress_get(ignore_connection_error=True)
    if cluster_update_progress.get('overall_status') != 'completed':
        cluster_update_progress = self.cluster_image_update_progress_get(ignore_connection_error=False)
    validation_reports = cluster_update_progress.get('validation_reports')
    if cluster_update_progress.get('overall_status') == 'completed':
        self.cluster_image_package_delete()
        return validation_reports
    if cluster_update_progress.get('overall_status') == 'in_progress':
        msg = 'Timeout error'
        action = '  Should the timeout value be increased?  Current value is %d seconds.' % self.parameters['timeout']
        action += '  The software update continues in background.'
    else:
        msg = 'Error'
        action = ''
    msg += ' updating image using ZAPI: overall_status: %s.' % cluster_update_progress.get('overall_status', 'cannot get status')
    msg += action
    self.module.fail_json(msg=msg, validation_reports=str(validation_reports), validation_reports_after_download=self.validation_reports_after_download, validation_reports_after_update=validation_reports)