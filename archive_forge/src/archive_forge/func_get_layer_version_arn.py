import base64
import hashlib
import re
import traceback
from collections import Counter
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_layer_version_arn(module, connection, layer_name, version_number):
    try:
        layer_versions = connection.list_layer_versions(LayerName=layer_name, aws_retry=True)['LayerVersions']
        for v in layer_versions:
            if v['Version'] == version_number:
                return v['LayerVersionArn']
        module.fail_json(msg=f'Unable to find version {version_number} from Lambda layer {layer_name}')
    except is_boto3_error_code('ResourceNotFoundException'):
        module.fail_json(msg=f'Lambda layer {layer_name} not found')