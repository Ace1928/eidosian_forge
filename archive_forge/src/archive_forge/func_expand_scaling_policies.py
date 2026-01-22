from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_scaling_policies(scaling_policies):
    eg_scaling_policies = []
    for policy in scaling_policies:
        eg_policy = expand_fields(scaling_policy_fields, policy, 'ScalingPolicy')
        eg_policy.action = expand_fields(action_fields, policy, 'ScalingPolicyAction')
        eg_scaling_policies.append(eg_policy)
    return eg_scaling_policies