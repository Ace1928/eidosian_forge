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
def paginated_versioned_list_with_fallback(s3, **pagination_params):
    try:
        versioned_pg = s3.get_paginator('list_object_versions')
        for page in versioned_pg.paginate(**pagination_params):
            delete_markers = [{'Key': data['Key'], 'VersionId': data['VersionId']} for data in page.get('DeleteMarkers', [])]
            current_objects = [{'Key': data['Key'], 'VersionId': data['VersionId']} for data in page.get('Versions', [])]
            yield (delete_markers + current_objects)
    except is_boto3_error_code(IGNORE_S3_DROP_IN_EXCEPTIONS + ['AccessDenied']):
        for key in paginated_list(s3, **pagination_params):
            yield [{'Key': key}]