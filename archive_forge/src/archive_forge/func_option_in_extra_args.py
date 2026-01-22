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
def option_in_extra_args(option):
    temp_option = option.replace('-', '').lower()
    allowed_extra_args = {'acl': 'ACL', 'cachecontrol': 'CacheControl', 'contentdisposition': 'ContentDisposition', 'contentencoding': 'ContentEncoding', 'contentlanguage': 'ContentLanguage', 'contenttype': 'ContentType', 'expires': 'Expires', 'grantfullcontrol': 'GrantFullControl', 'grantread': 'GrantRead', 'grantreadacp': 'GrantReadACP', 'grantwriteacp': 'GrantWriteACP', 'metadata': 'Metadata', 'requestpayer': 'RequestPayer', 'serversideencryption': 'ServerSideEncryption', 'storageclass': 'StorageClass', 'ssecustomeralgorithm': 'SSECustomerAlgorithm', 'ssecustomerkey': 'SSECustomerKey', 'ssecustomerkeymd5': 'SSECustomerKeyMD5', 'ssekmskeyid': 'SSEKMSKeyId', 'websiteredirectlocation': 'WebsiteRedirectLocation'}
    if temp_option in allowed_extra_args:
        return allowed_extra_args[temp_option]