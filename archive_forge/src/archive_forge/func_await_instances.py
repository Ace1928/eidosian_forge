import time
import uuid
from collections import namedtuple
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tower import tower_callback_script
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def await_instances(ids, desired_module_state='present', force_wait=False):
    if not module.params.get('wait', True) and (not force_wait):
        return
    if module.check_mode:
        return
    state_to_boto3_waiter = {'present': 'instance_exists', 'started': 'instance_status_ok', 'running': 'instance_running', 'stopped': 'instance_stopped', 'restarted': 'instance_status_ok', 'rebooted': 'instance_running', 'terminated': 'instance_terminated', 'absent': 'instance_terminated'}
    if desired_module_state not in state_to_boto3_waiter:
        module.fail_json(msg=f'Cannot wait for state {desired_module_state}, invalid state')
    boto3_waiter_type = state_to_boto3_waiter[desired_module_state]
    waiter = client.get_waiter(boto3_waiter_type)
    try:
        waiter.wait(InstanceIds=ids, WaiterConfig={'Delay': 15, 'MaxAttempts': module.params.get('wait_timeout', 600) // 15})
    except botocore.exceptions.WaiterConfigError as e:
        instance_ids = ', '.join(ids)
        module.fail_json(msg=f'{to_native(e)}. Error waiting for instances {instance_ids} to reach state {boto3_waiter_type}')
    except botocore.exceptions.WaiterError as e:
        instance_ids = ', '.join(ids)
        module.warn(f'Instances {instance_ids} took too long to reach state {boto3_waiter_type}. {to_native(e)}')