from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('protection_policies')
def generate_pp_dict(module, fusion):
    pp_info = {}
    api_instance = purefusion.ProtectionPoliciesApi(fusion)
    policies = api_instance.list_protection_policies()
    for policy in policies.items:
        policy_name = policy.name
        pp_info[policy_name] = {'objectives': policy.objectives}
    return pp_info