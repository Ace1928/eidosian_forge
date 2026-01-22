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
def wait_tags_are_applied(module, s3_client, bucket_name, expected_tags_dict):
    for dummy in range(0, 12):
        try:
            current_tags_dict = get_current_bucket_tags_dict(s3_client, bucket_name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to get bucket policy')
        if current_tags_dict != expected_tags_dict:
            time.sleep(5)
        else:
            return current_tags_dict
    module.fail_json(msg='Bucket tags failed to apply in the expected time', requested_tags=expected_tags_dict, live_tags=current_tags_dict)