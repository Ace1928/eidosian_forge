import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
@AWSRetry.jittered_backoff(**backoff_params)
def describe_autoscaling_groups(connection, group_name):
    pg = connection.get_paginator('describe_auto_scaling_groups')
    return pg.paginate(AutoScalingGroupNames=[group_name]).build_full_result().get('AutoScalingGroups', [])