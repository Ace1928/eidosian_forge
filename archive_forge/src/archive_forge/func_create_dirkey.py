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
def create_dirkey(module, s3, bucket, obj, encrypt, expiry):
    if module.check_mode:
        module.exit_json(msg='PUT operation skipped - running in check mode', changed=True)
    params = {'Bucket': bucket, 'Key': obj, 'Body': b''}
    params.update(get_extra_params(encrypt, module.params.get('encryption_mode'), module.params.get('encryption_kms_key_id')))
    put_object_acl(module, s3, bucket, obj, params)
    tags, _changed = ensure_tags(s3, module, bucket, obj)
    url = put_download_url(s3, bucket, obj, expiry)
    module.exit_json(msg=f'Virtual directory {obj} created in bucket {bucket}', url=url, tags=tags, changed=True)