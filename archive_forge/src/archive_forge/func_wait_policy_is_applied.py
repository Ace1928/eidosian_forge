import json
import time
from ansible.module_utils.basic import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def wait_policy_is_applied(module, s3_client, bucket_name, expected_policy, should_fail=True):
    for dummy in range(0, 12):
        try:
            current_policy = get_bucket_policy(s3_client, bucket_name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to get bucket policy')
        if compare_policies(current_policy, expected_policy):
            time.sleep(5)
        else:
            return current_policy
    if should_fail:
        module.fail_json(msg='Bucket policy failed to apply in the expected time', requested_policy=expected_policy, live_policy=current_policy)
    else:
        return None