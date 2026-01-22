import datetime
import itertools
import time
import uuid
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_stack_set(module, stack_params, cfn):
    try:
        cfn.create_stack_set(aws_retry=True, **stack_params)
        return await_stack_set_exists(cfn, stack_params['StackSetName'])
    except (ClientError, BotoCoreError) as err:
        module.fail_json_aws(err, msg=f'Failed to create stack set {stack_params.get('StackSetName')}.')