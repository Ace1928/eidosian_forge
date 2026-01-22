from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def post_update_tasks_rest(self, state):
    """delete software package when installation is successful
           report validation_results whether update succeeded or failed
        """
    validation_reports, error = self.cluster_image_get_rest('validation_results', fail_on_error=False)
    if state == 'completed':
        self.cluster_image_package_delete()
        return error or validation_reports
    if state == 'in_progress':
        msg = 'Timeout error'
        action = '  Should the timeout value be increased?  Current value is %d seconds.' % self.parameters['timeout']
        action += '  The software update continues in background.'
    else:
        msg = 'Error'
        action = ''
    msg += ' updating image using REST: state: %s.' % state
    msg += action
    self.module.fail_json(msg=msg, validation_reports_after_download=self.validation_reports_after_download, validation_reports_after_update=error or validation_reports)