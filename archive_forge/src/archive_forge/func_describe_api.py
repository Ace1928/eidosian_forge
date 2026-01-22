import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**retry_params)
def describe_api(client, module, rest_api_id):
    try:
        response = client.get_rest_api(restApiId=rest_api_id)
        response.pop('ResponseMetadata')
    except is_boto3_error_code('ResourceNotFoundException'):
        response = {}
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f"Trying to get Rest API '{rest_api_id}'.")
    return response