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
def normalize_iam_instance_profile(profile, _v7_compat=False):
    """
    Converts a boto3 format IAM instance profile into "Ansible" format

    _v7_compat is deprecated and will be removed in release after 2025-05-01 DO NOT USE.
    """
    new_profile = camel_dict_to_snake_dict(deepcopy(profile))
    if profile.get('Roles'):
        new_profile['roles'] = [normalize_iam_role(role, _v7_compat=_v7_compat) for role in profile.get('Roles')]
    if profile.get('Tags'):
        new_profile['tags'] = boto3_tag_list_to_ansible_dict(profile.get('Tags'))
    else:
        new_profile['tags'] = {}
    new_profile['original'] = profile
    return new_profile