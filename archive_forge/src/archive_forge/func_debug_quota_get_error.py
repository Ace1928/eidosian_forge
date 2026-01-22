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
def debug_quota_get_error(self, error):
    policies = self.get_quota_policies()
    entries = {}
    for policy in policies:
        entries[policy] = self.get_quotas(policy)
    if len(policies) == 1:
        self.module.warn('retried with success using policy="%s" on "13001:success" ZAPI error.' % policy)
        return entries[policies[0]]
    self.module.fail_json(msg='Error fetching quotas info: %s - current vserver policies: %s, details: %s' % (to_native(error), policies, entries))