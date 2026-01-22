from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=10, delay=10, catch_extra_error_codes=['TargetGroupNotFound'])
def describe_target_groups_with_backoff(connection, tg_name):
    return connection.describe_target_groups(Names=[tg_name])