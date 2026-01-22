import json
import time
import traceback
import uuid
from hashlib import sha1
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def update_termination_protection(module, cfn, stack_name, desired_termination_protection_state):
    """updates termination protection of a stack"""
    stack = get_stack_facts(module, cfn, stack_name)
    if stack:
        if stack['EnableTerminationProtection'] is not desired_termination_protection_state:
            try:
                cfn.update_termination_protection(aws_retry=True, EnableTerminationProtection=desired_termination_protection_state, StackName=stack_name)
            except botocore.exceptions.ClientError as e:
                module.fail_json_aws(e)