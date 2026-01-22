from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import re
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.core import log
def get_bucket_resource_from_xml_response(scheme, bucket_dict, bucket_name):
    """Creates resource_reference.S3BucketResource from S3 API response.

  Args:
    scheme (storage_url.ProviderPrefix): Prefix used for provider URLs.
    bucket_dict (dict): Dictionary representing S3 API response.
    bucket_name (str): Bucket response is relevant to.

  Returns:
    resource_reference.S3BucketResource populated with data.
  """
    requester_pays = _get_error_or_value(bucket_dict.get('Payer'))
    if requester_pays == 'Requester':
        requester_pays = True
    elif requester_pays == 'BucketOwner':
        requester_pays = False
    versioning_enabled = _get_error_or_value(bucket_dict.get('Versioning'))
    if isinstance(versioning_enabled, dict):
        if versioning_enabled.get('Status') == 'Enabled':
            versioning_enabled = True
        else:
            versioning_enabled = None
    return _SCHEME_TO_BUCKET_RESOURCE_DICT[scheme](storage_url.CloudUrl(scheme, bucket_name), acl=_get_error_or_value(bucket_dict.get('ACL')), cors_config=_get_error_or_value(bucket_dict.get('CORSRules')), lifecycle_config=_get_error_or_value(bucket_dict.get('LifecycleConfiguration')), logging_config=_get_error_or_value(bucket_dict.get('LoggingEnabled')), requester_pays=requester_pays, location=_get_error_or_value(bucket_dict.get('LocationConstraint')), metadata=bucket_dict, versioning_enabled=versioning_enabled, website_config=_get_error_or_value(bucket_dict.get('Website')))