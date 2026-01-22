from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_launch_spec(eg_compute, module, is_update, do_not_update):
    eg_launch_spec = expand_fields(lspec_fields, module.params, 'LaunchSpecification')
    if module.params['iam_role_arn'] is not None or module.params['iam_role_name'] is not None:
        eg_launch_spec.iam_role = expand_fields(iam_fields, module.params, 'IamRole')
    tags = module.params['tags']
    load_balancers = module.params['load_balancers']
    target_group_arns = module.params['target_group_arns']
    block_device_mappings = module.params['block_device_mappings']
    network_interfaces = module.params['network_interfaces']
    if is_update is True:
        if 'image_id' in do_not_update:
            delattr(eg_launch_spec, 'image_id')
    expand_tags(eg_launch_spec, tags)
    expand_load_balancers(eg_launch_spec, load_balancers, target_group_arns)
    expand_block_device_mappings(eg_launch_spec, block_device_mappings)
    expand_network_interfaces(eg_launch_spec, network_interfaces)
    eg_compute.launch_specification = eg_launch_spec