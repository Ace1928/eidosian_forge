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
def update_object_metadata_dict_from_request_config(object_metadata, request_config, attributes_resource=None, posix_to_set=None):
    """Returns S3 object metadata dict fields based on RequestConfig.

  Args:
    object_metadata (dict): Existing object metadata.
    request_config (request_config): May contain data to add to object_metadata.
    attributes_resource (Resource|None): If present, used for parsing POSIX and
      symlink data from a resource for the --preserve-posix and/or
      --preserve_symlink flags. This value is ignored unless it is an instance
      of FileObjectResource.
    posix_to_set (PosixAttributes|None): Set as custom metadata on target.

  """
    if request_config.predefined_acl_string is not None:
        object_metadata['ACL'] = translate_predefined_acl_string_to_s3(request_config.predefined_acl_string)
    resource_args = request_config.resource_args
    existing_metadata = object_metadata.get('Metadata', {})
    custom_fields_dict = metadata_util.get_updated_custom_fields(existing_metadata, request_config, attributes_resource=attributes_resource, known_posix=posix_to_set)
    if custom_fields_dict is not None:
        object_metadata['Metadata'] = custom_fields_dict
    if resource_args:
        for field, value in _S3_TO_GENERIC_FIELD_NAMES_.items():
            _process_value_or_clear_flag(object_metadata, field, getattr(resource_args, value, None))