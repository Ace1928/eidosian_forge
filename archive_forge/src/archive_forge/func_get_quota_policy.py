from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import zapis_svm
def get_quota_policy(self, policy_name=None):
    if policy_name is None:
        policy_name = self.parameters['name']
    return_value = None
    quota_policy_get_iter = netapp_utils.zapi.NaElement('quota-policy-get-iter')
    quota_policy_info = netapp_utils.zapi.NaElement('quota-policy-info')
    quota_policy_info.add_new_child('policy-name', policy_name)
    quota_policy_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(quota_policy_info)
    quota_policy_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(quota_policy_get_iter, True)
        if result.get_child_by_name('attributes-list'):
            quota_policy_attributes = result['attributes-list']['quota-policy-info']
            return_value = {'name': quota_policy_attributes['policy-name']}
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching quota policy %s: %s' % (policy_name, to_native(error)), exception=traceback.format_exc())
    return return_value