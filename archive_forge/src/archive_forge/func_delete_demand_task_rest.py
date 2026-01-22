from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_demand_task_rest(self):
    api = 'protocols/vscan/%s/on-demand-policies' % self.svm_uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['task_name'])
    if error:
        self.module.fail_json(msg='Error deleting on demand task %s: %s' % (self.parameters['task_name'], to_native(error)))