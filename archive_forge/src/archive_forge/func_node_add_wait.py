from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def node_add_wait(self):
    """
        Wait whilst node is being added to the existing cluster
        """
    if self.use_rest:
        return
    cluster_node_status = netapp_utils.zapi.NaElement('cluster-add-node-status-get-iter')
    node_status_info = netapp_utils.zapi.NaElement('cluster-create-add-node-status-info')
    node_status_info.add_new_child('cluster-ip', self.parameters.get('cluster_ip_address'))
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(node_status_info)
    cluster_node_status.add_child_elem(query)
    is_complete = None
    failure_msg = None
    retries = self.parameters['time_out']
    errors = []
    while is_complete != 'success' and is_complete != 'failure' and (retries > 0):
        retries = retries - 10
        time.sleep(10)
        try:
            result = self.server.invoke_successfully(cluster_node_status, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            if error.message == 'Unable to find API: cluster-add-node-status-get-iter':
                time.sleep(60)
                return
            errors.append(repr(error))
            continue
        attributes_list = result.get_child_by_name('attributes-list')
        join_progress = attributes_list.get_child_by_name('cluster-create-add-node-status-info')
        is_complete = join_progress.get_child_content('status')
        failure_msg = join_progress.get_child_content('failure-msg')
    if self.parameters['time_out'] == 0:
        is_complete = 'success'
    if is_complete != 'success':
        if 'Node is already in a cluster' in failure_msg:
            return
        elif retries <= 0:
            errors.append('Timeout after %s seconds' % self.parameters['time_out'])
        if failure_msg:
            errors.append(failure_msg)
        self.module.fail_json(msg='Error adding node with ip address %s: %s' % (self.parameters['cluster_ip_address'], str(errors)))