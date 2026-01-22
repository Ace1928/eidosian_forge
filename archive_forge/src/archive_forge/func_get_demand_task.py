from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_demand_task(self):
    """
        Get a demand task
        :return: A vscan-on-demand-task-info or None
        """
    if self.use_rest:
        self.get_svm_uuid()
        return self.get_demand_task_rest()
    demand_task_iter = netapp_utils.zapi.NaElement('vscan-on-demand-task-get-iter')
    demand_task_info = netapp_utils.zapi.NaElement('vscan-on-demand-task-info')
    demand_task_info.add_new_child('task-name', self.parameters['task_name'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(demand_task_info)
    demand_task_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(demand_task_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error searching for Vscan on demand task %s: %s' % (self.parameters['task_name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return result.get_child_by_name('attributes-list').get_child_by_name('vscan-on-demand-task-info')
    return None