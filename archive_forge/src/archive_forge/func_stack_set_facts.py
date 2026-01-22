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
@AWSRetry.jittered_backoff(retries=3, delay=4)
def stack_set_facts(cfn, stack_set_name):
    try:
        ss = cfn.describe_stack_set(StackSetName=stack_set_name)['StackSet']
        ss['Tags'] = boto3_tag_list_to_ansible_dict(ss['Tags'])
        return ss
    except cfn.exceptions.from_code('StackSetNotFound'):
        return