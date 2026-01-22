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
def upload_s3file(module, s3, bucket, obj, expiry, metadata, encrypt, headers, src=None, content=None, acl_disabled=False):
    if module.check_mode:
        module.exit_json(msg='PUT operation skipped - running in check mode', changed=True)
    try:
        extra = get_extra_params(encrypt, module.params.get('encryption_mode'), module.params.get('encryption_kms_key_id'), metadata)
        if module.params.get('permission'):
            permissions = module.params['permission']
            if isinstance(permissions, str):
                extra['ACL'] = permissions
            elif isinstance(permissions, list):
                extra['ACL'] = permissions[0]
        if 'ContentType' not in extra:
            extra['ContentType'] = guess_content_type(src)
        if src:
            s3.upload_file(aws_retry=True, Filename=src, Bucket=bucket, Key=obj, ExtraArgs=extra)
        else:
            f = io.BytesIO(content)
            s3.upload_fileobj(aws_retry=True, Fileobj=f, Bucket=bucket, Key=obj, ExtraArgs=extra)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError, boto3.exceptions.Boto3Error) as e:
        raise S3ObjectFailure('Unable to complete PUT operation.', e)
    if not acl_disabled:
        put_object_acl(module, s3, bucket, obj)
    tags, _changed = ensure_tags(s3, module, bucket, obj)
    url = put_download_url(s3, bucket, obj, expiry)
    module.exit_json(msg='PUT operation complete', url=url, tags=tags, changed=True)