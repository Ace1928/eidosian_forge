import base64
import copy
import io
import mimetypes
import os
import time
from ssl import SSLError
from ansible.module_utils.basic import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import HAS_MD5
from ansible_collections.amazon.aws.plugins.module_utils.s3 import calculate_etag
from ansible_collections.amazon.aws.plugins.module_utils.s3 import calculate_etag_content
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_extra_params(encrypt=None, encryption_mode=None, encryption_kms_key_id=None, metadata=None):
    extra = {}
    if encrypt:
        extra['ServerSideEncryption'] = encryption_mode
    if encryption_kms_key_id and encryption_mode == 'aws:kms':
        extra['SSEKMSKeyId'] = encryption_kms_key_id
    if metadata:
        extra['Metadata'] = {}
        for option in metadata:
            extra_args_option = option_in_extra_args(option)
            if extra_args_option:
                extra[extra_args_option] = metadata[option]
            else:
                extra['Metadata'][option] = metadata[option]
    return extra