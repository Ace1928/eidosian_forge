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
def compare_stack_instances(cfn, stack_set_name, accounts, regions):
    instance_list = cfn.list_stack_instances(aws_retry=True, StackSetName=stack_set_name)['Summaries']
    desired_stack_instances = set(itertools.product(accounts, regions))
    existing_stack_instances = set(((i['Account'], i['Region']) for i in instance_list))
    return (desired_stack_instances - existing_stack_instances, existing_stack_instances, existing_stack_instances - desired_stack_instances)