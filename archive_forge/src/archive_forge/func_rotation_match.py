import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def rotation_match(desired_secret, current_secret):
    """Compare secrets rotation configuration

    Args:
        desired_secret: camel dict representation of the desired secret state.
        current_secret: secret reference as returned by the secretsmanager api.

    Returns: bool
    """
    if desired_secret.rotation_enabled != current_secret.get('RotationEnabled', False):
        return False
    if desired_secret.rotation_enabled:
        if desired_secret.rotation_lambda_arn != current_secret.get('RotationLambdaARN'):
            return False
        if desired_secret.rotation_rules != current_secret.get('RotationRules'):
            return False
    return True