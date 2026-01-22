from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_create_wait(self):
    """
        Wait whilst cluster creation completes
        """
    if self.use_rest:
        return
    cluster_wait = netapp_utils.zapi.NaElement('cluster-create-join-progress-get')
    is_complete = False
    status = ''
    retries = self.parameters['time_out']
    errors = []
    while not is_complete and status not in ('failed', 'success') and (retries > 0):
        retries = retries - 10
        time.sleep(10)
        try:
            result = self.server.invoke_successfully(cluster_wait, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            errors.append(repr(error))
            continue
        clus_progress = result.get_child_by_name('attributes')
        result = clus_progress.get_child_by_name('cluster-create-join-progress-info')
        is_complete = self.na_helper.get_value_for_bool(from_zapi=True, value=result.get_child_content('is-complete'))
        status = result.get_child_content('status')
    if self.parameters['time_out'] == 0:
        is_complete = True
    if not is_complete and status != 'success':
        current_status_message = result.get_child_content('current-status-message')
        errors.append('Failed to confirm cluster creation %s: %s' % (self.parameters.get('cluster_name'), current_status_message))
        if retries <= 0:
            errors.append('Timeout after %s seconds' % self.parameters['time_out'])
        self.module.fail_json(msg='Error creating cluster %s: %s' % (self.parameters['cluster_name'], str(errors)))
    return is_complete