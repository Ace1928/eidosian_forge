import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**retry_params)
def create_api(client, name, description=None, endpoint_type=None, tags=None):
    params = {'name': name}
    if description:
        params['description'] = description
    if endpoint_type:
        params['endpointConfiguration'] = {'types': [endpoint_type]}
    if tags:
        params['tags'] = tags
    return client.create_rest_api(**params)