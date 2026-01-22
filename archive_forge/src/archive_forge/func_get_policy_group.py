from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_policy_group(self, policy_group_name=None):
    """
        Return details of a policy group.
        :param policy_group_name: policy group name
        :return: policy group details.
        :rtype: dict.
        """
    if policy_group_name is None:
        policy_group_name = self.parameters['name']
    policy_group_get_iter = netapp_utils.zapi.NaElement('qos-adaptive-policy-group-get-iter')
    policy_group_info = netapp_utils.zapi.NaElement('qos-adaptive-policy-group-info')
    policy_group_info.add_new_child('policy-group', policy_group_name)
    policy_group_info.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(policy_group_info)
    policy_group_get_iter.add_child_elem(query)
    result = self.server.invoke_successfully(policy_group_get_iter, True)
    policy_group_detail = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        policy_info = result.get_child_by_name('attributes-list').get_child_by_name('qos-adaptive-policy-group-info')
        policy_group_detail = {'name': policy_info.get_child_content('policy-group'), 'vserver': policy_info.get_child_content('vserver'), 'absolute_min_iops': policy_info.get_child_content('absolute-min-iops'), 'expected_iops': policy_info.get_child_content('expected-iops'), 'peak_iops': policy_info.get_child_content('peak-iops'), 'peak_iops_allocation': policy_info.get_child_content('peak-iops-allocation')}
    return policy_group_detail