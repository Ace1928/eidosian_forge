import os
import uuid
from ansible.module_utils._text import to_bytes
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def create_new_key_pair(ec2_client, name, key_material, key_type, tags, file_name, check_mode):
    """
    key does not exist, we create new key
    """
    if check_mode:
        return {'changed': True, 'key': None, 'msg': 'key pair created'}
    tag_spec = boto3_tag_specifications(tags, ['key-pair'])
    if key_material:
        key = _import_key_pair(ec2_client, name, key_material, tag_spec)
    else:
        key = _create_key_pair(ec2_client, name, tag_spec, key_type)
    key_data = extract_key_data(key, key_type, file_name)
    result = {'changed': True, 'key': key_data, 'msg': 'key pair created'}
    return result