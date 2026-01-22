import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def try_except_ClientError(failure_msg):
    """
    Wrapper for boto3 calls that uses AWSRetry and handles exceptions
    """

    def wrapper(f):

        def run_func(*args, **kwargs):
            try:
                result = AWSRetry.jittered_backoff(retries=8, delay=5, catch_extra_error_codes=['DirectConnectClientException'])(f)(*args, **kwargs)
            except (ClientError, BotoCoreError) as e:
                raise DirectConnectError(failure_msg, traceback.format_exc(), e)
            return result
        return run_func
    return wrapper