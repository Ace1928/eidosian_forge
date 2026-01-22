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
def get_current_object_tags_dict(module, s3, bucket, obj, version=None):
    try:
        if version:
            current_tags = s3.get_object_tagging(aws_retry=True, Bucket=bucket, Key=obj, VersionId=version).get('TagSet')
        else:
            current_tags = s3.get_object_tagging(aws_retry=True, Bucket=bucket, Key=obj).get('TagSet')
    except is_boto3_error_code(IGNORE_S3_DROP_IN_EXCEPTIONS):
        module.warn('GetObjectTagging is not implemented by your storage provider.')
        return {}
    except is_boto3_error_code(['NoSuchTagSet', 'NoSuchTagSetError']):
        return {}
    return boto3_tag_list_to_ansible_dict(current_tags)