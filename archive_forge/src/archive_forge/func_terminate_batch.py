import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def terminate_batch(connection, replace_instances, initial_instances, leftovers=False):
    batch_size = module.params.get('replace_batch_size')
    min_size = module.params.get('min_size')
    desired_capacity = module.params.get('desired_capacity')
    group_name = module.params.get('name')
    lc_check = module.params.get('lc_check')
    lt_check = module.params.get('lt_check')
    decrement_capacity = False
    break_loop = False
    as_group = describe_autoscaling_groups(connection, group_name)[0]
    if desired_capacity is None:
        desired_capacity = as_group['DesiredCapacity']
    props = get_properties(as_group)
    desired_size = as_group['MinSize']
    if module.params.get('launch_config_name'):
        new_instances, old_instances = get_instances_by_launch_config(props, lc_check, initial_instances)
    else:
        new_instances, old_instances = get_instances_by_launch_template(props, lt_check, initial_instances)
    num_new_inst_needed = desired_capacity - len(new_instances)
    instances_to_terminate = list_purgeable_instances(props, lc_check, lt_check, replace_instances, initial_instances)
    module.debug(f'new instances needed: {num_new_inst_needed}')
    module.debug(f'new instances: {(*new_instances,)}')
    module.debug(f'old instances: {(*old_instances,)}')
    module.debug(f'batch instances: {(*instances_to_terminate,)}')
    if num_new_inst_needed == 0:
        decrement_capacity = True
        if as_group['MinSize'] != min_size:
            if min_size is None:
                min_size = as_group['MinSize']
            updated_params = dict(AutoScalingGroupName=as_group['AutoScalingGroupName'], MinSize=min_size)
            update_asg(connection, **updated_params)
            module.debug(f'Updating minimum size back to original of {min_size}')
        if leftovers:
            decrement_capacity = False
        break_loop = True
        instances_to_terminate = old_instances
        desired_size = min_size
        module.debug('No new instances needed')
    if num_new_inst_needed < batch_size and num_new_inst_needed != 0:
        instances_to_terminate = instances_to_terminate[:num_new_inst_needed]
        decrement_capacity = False
        break_loop = False
        module.debug(f'{num_new_inst_needed} new instances needed')
    module.debug(f'decrementing capacity: {decrement_capacity}')
    for instance_id in instances_to_terminate:
        elb_dreg(connection, group_name, instance_id)
        module.debug(f'terminating instance: {instance_id}')
        terminate_asg_instance(connection, instance_id, decrement_capacity)
    return (break_loop, desired_size, instances_to_terminate)