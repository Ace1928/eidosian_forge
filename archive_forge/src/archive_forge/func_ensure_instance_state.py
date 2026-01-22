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
def ensure_instance_state(desired_module_state, filters):
    """
    Sets return keys depending on the desired instance state
    """
    results = dict()
    changed = False
    if desired_module_state in ('running', 'started'):
        _changed, failed, instances, failure_reason = change_instance_state(filters=filters, desired_module_state=desired_module_state)
        changed |= bool(len(_changed))
        if failed:
            module.fail_json(msg=f'Unable to start instances: {failure_reason}', reboot_success=list(_changed), reboot_failed=failed)
        results = dict(msg='Instances started', start_success=list(_changed), start_failed=[], reboot_success=list(_changed), reboot_failed=[], changed=changed, instances=[pretty_instance(i) for i in instances])
    elif desired_module_state in ('restarted', 'rebooted'):
        _changed, failed, instances, failure_reason = change_instance_state(filters=filters, desired_module_state='stopped')
        if failed:
            module.fail_json(msg=f'Unable to stop instances: {failure_reason}', stop_success=list(_changed), stop_failed=failed)
        changed |= bool(len(_changed))
        _changed, failed, instances, failure_reason = change_instance_state(filters=filters, desired_module_state=desired_module_state)
        changed |= bool(len(_changed))
        if failed:
            module.fail_json(msg=f'Unable to restart instances: {failure_reason}', reboot_success=list(_changed), reboot_failed=failed)
        results = dict(msg='Instances restarted', reboot_success=list(_changed), changed=changed, reboot_failed=[], instances=[pretty_instance(i) for i in instances])
    elif desired_module_state in ('stopped',):
        _changed, failed, instances, failure_reason = change_instance_state(filters=filters, desired_module_state=desired_module_state)
        changed |= bool(len(_changed))
        if failed:
            module.fail_json(msg=f'Unable to stop instances: {failure_reason}', stop_success=list(_changed), stop_failed=failed)
        results = dict(msg='Instances stopped', stop_success=list(_changed), changed=changed, stop_failed=[], instances=[pretty_instance(i) for i in instances])
    elif desired_module_state in ('absent', 'terminated'):
        terminated, terminate_failed, instances, failure_reason = change_instance_state(filters=filters, desired_module_state=desired_module_state)
        if terminate_failed:
            module.fail_json(msg=f'Unable to terminate instances: {failure_reason}', terminate_success=list(terminated), terminate_failed=terminate_failed)
        results = dict(msg='Instances terminated', terminate_success=list(terminated), changed=bool(len(terminated)), terminate_failed=[], instances=[pretty_instance(i) for i in instances])
    return results