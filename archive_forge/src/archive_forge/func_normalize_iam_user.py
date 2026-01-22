import re
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .arn import parse_aws_arn
from .arn import validate_aws_arn
from .botocore import is_boto3_error_code
from .botocore import normalize_boto3_result
from .errors import AWSErrorHandler
from .exceptions import AnsibleAWSError
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def normalize_iam_user(user):
    """Converts IAM users from the CamelCase boto3 format to the snake_case Ansible format"""
    if not user:
        return user
    camel_user = camel_dict_to_snake_dict(user)
    camel_user['tags'] = boto3_tag_list_to_ansible_dict(user.pop('Tags', []))
    return camel_user