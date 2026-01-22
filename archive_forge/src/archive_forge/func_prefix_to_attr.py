from collections import defaultdict
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prefix_to_attr(attr_id):
    """
    Helper method to convert ID prefix to mount target attribute
    """
    attr_by_prefix = {'fsmt-': 'mount_target_id', 'subnet-': 'subnet_id', 'eni-': 'network_interface_id', 'sg-': 'security_groups'}
    return first_or_default([attr_name for prefix, attr_name in attr_by_prefix.items() if str(attr_id).startswith(prefix)], 'ip_address')