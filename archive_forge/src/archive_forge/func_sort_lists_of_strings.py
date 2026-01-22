import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def sort_lists_of_strings(policy):
    for statement_index in range(0, len(policy.get('Statement', []))):
        for key in policy['Statement'][statement_index]:
            value = policy['Statement'][statement_index][key]
            if isinstance(value, list) and all((isinstance(item, string_types) for item in value)):
                policy['Statement'][statement_index][key] = sorted(value)
    return policy