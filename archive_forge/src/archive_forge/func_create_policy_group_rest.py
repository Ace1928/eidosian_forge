from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_policy_group_rest(self):
    api = 'storage/qos/policies'
    body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
    if 'fixed_qos_options' in self.parameters:
        body['fixed'] = self.na_helper.filter_out_none_entries(self.parameters['fixed_qos_options'])
        if self.na_helper.safe_get(body, ['fixed', 'capacity_shared']) is None:
            body['fixed']['capacity_shared'] = False
    else:
        body['adaptive'] = self.na_helper.filter_out_none_entries(self.parameters['adaptive_qos_options'])
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating qos policy group %s: %s' % (self.parameters['name'], error))