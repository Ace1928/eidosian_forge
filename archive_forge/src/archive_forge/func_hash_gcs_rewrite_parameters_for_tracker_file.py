from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def hash_gcs_rewrite_parameters_for_tracker_file(source_object_resource, destination_object_resource, destination_metadata=None, request_config=None):
    """Creates an MD5 hex digest of the parameters for GCS rewrite call.

  Resuming rewrites requires that the input parameters are identical, so the
  tracker file needs to represent the input parameters. This is done by hashing
  the API call parameters. For example, if a user performs a rewrite with a
  changed ACL, the hashes will not match, and we will restart the rewrite.

  Args:
    source_object_resource (ObjectResource): Must include bucket, name, etag,
      and metadata.
    destination_object_resource (ObjectResource|UnknownResource): Must include
      bucket, name, and metadata.
    destination_metadata (messages.Object|None): Separated from
      destination_object_resource since UnknownResource does not have metadata.
    request_config (request_config_factory._RequestConfig|None): Contains a
      variety of API arguments.

  Returns:
    MD5 hex digest (string) of the input parameters.

  Raises:
    Error if argument is missing required property.
  """
    mandatory_parameters = (source_object_resource.storage_url.bucket_name, source_object_resource.storage_url.object_name, source_object_resource.etag, destination_object_resource.storage_url.bucket_name, destination_object_resource.storage_url.object_name)
    if not all(mandatory_parameters):
        raise errors.Error('Missing required parameter values.')
    source_encryption = source_object_resource.decryption_key_hash_sha256 or source_object_resource.kms_key
    destination_encryption = None
    if request_config and request_config.resource_args and isinstance(request_config.resource_args.encryption_key, encryption_util.EncryptionKey):
        key = request_config.resource_args.encryption_key
        if key.type is encryption_util.KeyType.CSEK:
            destination_encryption = key.sha256
        elif key.type is encryption_util.KeyType.CMEK:
            destination_encryption = key.key
    optional_parameters = (destination_metadata, scaled_integer.ParseInteger(properties.VALUES.storage.copy_chunk_size.Get()), getattr(request_config, 'precondition_generation_match', None), getattr(request_config, 'precondition_metageneration_match', None), getattr(request_config, 'predefined_acl_string', None), source_encryption, destination_encryption)
    all_parameters = mandatory_parameters + optional_parameters
    parameters_bytes = ''.join([str(parameter) for parameter in all_parameters]).encode('UTF8')
    parameters_hash = hashing.get_md5(parameters_bytes)
    return parameters_hash.hexdigest()