from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def get_cleared_object_fields(request_config):
    """Gets a list of fields to be included in requests despite null values."""
    cleared_fields = []
    resource_args = request_config.resource_args
    if not resource_args:
        return cleared_fields
    if resource_args.cache_control == user_request_args_factory.CLEAR:
        cleared_fields.append('cacheControl')
    if resource_args.content_disposition == user_request_args_factory.CLEAR:
        cleared_fields.append('contentDisposition')
    if resource_args.content_encoding == user_request_args_factory.CLEAR:
        cleared_fields.append('contentEncoding')
    if resource_args.content_language == user_request_args_factory.CLEAR:
        cleared_fields.append('contentLanguage')
    if resource_args.custom_time == user_request_args_factory.CLEAR:
        cleared_fields.append('customTime')
    if resource_args.retain_until == user_request_args_factory.CLEAR or resource_args.retention_mode == user_request_args_factory.CLEAR:
        cleared_fields.append('retention')
    return cleared_fields