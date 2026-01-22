from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_quota_policies(self):
    """
        Get list of quota policies
        :return: list of quota policies (empty list if None found)
        """
    quota_policy_get = netapp_utils.zapi.NaElement('quota-policy-get-iter')
    query = {'query': {'quota-policy-info': {'vserver': self.parameters['vserver']}}}
    quota_policy_get.translate_struct(query)
    try:
        result = self.server.invoke_successfully(quota_policy_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching quota policies: %s' % to_native(error), exception=traceback.format_exc())
    return [policy['policy-name'] for policy in result['attributes-list'].get_children()] if result.get_child_by_name('attributes-list') else []