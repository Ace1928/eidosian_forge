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
def s3_object_do_getstr(module, connection, connection_v4, s3_vars):
    if module.params.get('sig_v4'):
        connection = connection_v4
    if s3_vars['bucket'] and s3_vars['object']:
        if key_check(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'], validate=s3_vars['validate']):
            try:
                download_s3str(module, connection, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'])
            except Sigv4Required:
                download_s3str(module, connection_v4, s3_vars['bucket'], s3_vars['object'], version=s3_vars['version'])
        elif s3_vars['version']:
            module.fail_json(msg=f'Key {s3_vars['object']} with version id {s3_vars['version']} does not exist.')
        else:
            module.fail_json(msg=f'Key {s3_vars['object']} does not exist.')