import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_api_definitions(module, swagger_file=None, swagger_dict=None, swagger_text=None):
    apidata = None
    if swagger_file is not None:
        try:
            with open(swagger_file) as f:
                apidata = f.read()
        except OSError as e:
            msg = f'Failed trying to read swagger file {str(swagger_file)}: {str(e)}'
            module.fail_json(msg=msg, exception=traceback.format_exc())
    if swagger_dict is not None:
        apidata = json.dumps(swagger_dict)
    if swagger_text is not None:
        apidata = swagger_text
    if apidata is None:
        module.fail_json(msg='module error - no swagger info provided')
    return apidata