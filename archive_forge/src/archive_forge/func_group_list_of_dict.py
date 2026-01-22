from collections import defaultdict
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def group_list_of_dict(array):
    """
    Helper method to group list of dict to dict with all possible values
    """
    result = defaultdict(list)
    for item in array:
        for key, value in item.items():
            result[key] += value if isinstance(value, list) else [value]
    return result