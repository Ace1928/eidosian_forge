from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_scaling(eg, module):
    up_scaling_policies = module.params['up_scaling_policies']
    down_scaling_policies = module.params['down_scaling_policies']
    target_tracking_policies = module.params['target_tracking_policies']
    eg_scaling = spotinst.aws_elastigroup.Scaling()
    if up_scaling_policies is not None:
        eg_up_scaling_policies = expand_scaling_policies(up_scaling_policies)
        if len(eg_up_scaling_policies) > 0:
            eg_scaling.up = eg_up_scaling_policies
    if down_scaling_policies is not None:
        eg_down_scaling_policies = expand_scaling_policies(down_scaling_policies)
        if len(eg_down_scaling_policies) > 0:
            eg_scaling.down = eg_down_scaling_policies
    if target_tracking_policies is not None:
        eg_target_tracking_policies = expand_target_tracking_policies(target_tracking_policies)
        if len(eg_target_tracking_policies) > 0:
            eg_scaling.target = eg_target_tracking_policies
    if eg_scaling.down is not None or eg_scaling.up is not None or eg_scaling.target is not None:
        eg.scaling = eg_scaling