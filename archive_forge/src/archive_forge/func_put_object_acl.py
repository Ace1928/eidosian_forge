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
def put_object_acl(module, s3, bucket, obj, params=None):
    try:
        if params:
            s3.put_object(aws_retry=True, **params)
        for acl in module.params.get('permission'):
            s3.put_object_acl(aws_retry=True, ACL=acl, Bucket=bucket, Key=obj)
    except is_boto3_error_code(IGNORE_S3_DROP_IN_EXCEPTIONS):
        module.warn('PutObjectAcl is not implemented by your storage provider. Set the permissions parameters to the empty list to avoid this warning')
    except is_boto3_error_code('AccessControlListNotSupported'):
        module.warn('PutObjectAcl operation : The bucket does not allow ACLs.')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError, boto3.exceptions.Boto3Error) as e:
        raise S3ObjectFailure(f'Failed while creating object {obj}.', e)