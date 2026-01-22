import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_api_by_tags(client, module, name, tags):
    count = 0
    result = None
    for api in list_apis(client):
        if name and api['name'] != name:
            continue
        api_tags = api.get('tags', {})
        if all((tag_key in api_tags and api_tags[tag_key] == tag_value for tag_key, tag_value in tags.items())):
            result = api
            count += 1
    if count > 1:
        args = 'Tags'
        if name:
            args += ' and name'
        module.fail_json(msg=f'{args} provided do not identify a unique API gateway')
    return result