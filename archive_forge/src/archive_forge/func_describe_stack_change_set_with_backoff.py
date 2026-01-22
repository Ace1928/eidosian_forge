import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
@AWSRetry.exponential_backoff(retries=5, delay=5)
def describe_stack_change_set_with_backoff(self, **kwargs):
    paginator = self.client.get_paginator('describe_change_set')
    return paginator.paginate(**kwargs).build_full_result()