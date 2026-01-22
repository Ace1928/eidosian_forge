from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
def get_request_config(url, content_type=None, decryption_key_hash_sha256=None, encryption_key=None, error_on_missing_key=True, md5_hash=None, size=None, user_request_args=None):
    """Generates API-specific RequestConfig. See output classes for arg info."""
    resource_args = _get_request_config_resource_args(url, content_type, decryption_key_hash_sha256, encryption_key, error_on_missing_key, md5_hash, size, user_request_args)
    if url.scheme == storage_url.ProviderPrefix.GCS:
        request_config = _GcsRequestConfig(resource_args=resource_args)
        if user_request_args:
            request_config.gzip_settings = user_request_args.gzip_settings
            request_config.override_unlocked_retention = user_request_args.override_unlocked_retention
            if user_request_args.no_clobber:
                request_config.no_clobber = user_request_args.no_clobber
            if user_request_args.precondition_generation_match:
                request_config.precondition_generation_match = int(user_request_args.precondition_generation_match)
            if user_request_args.precondition_metageneration_match:
                request_config.precondition_metageneration_match = int(user_request_args.precondition_metageneration_match)
    elif url.scheme == storage_url.ProviderPrefix.S3:
        request_config = _S3RequestConfig(resource_args=resource_args)
    else:
        request_config = _RequestConfig(resource_args=resource_args)
    request_config.default_object_acl_file_path = getattr(user_request_args, 'default_object_acl_file_path', None)
    request_config.predefined_acl_string = getattr(user_request_args, 'predefined_acl_string', None)
    request_config.predefined_default_object_acl_string = getattr(user_request_args, 'predefined_default_object_acl_string', None)
    request_config.preserve_posix = getattr(user_request_args, 'preserve_posix', None)
    request_config.preserve_symlinks = user_request_args.preserve_symlinks if user_request_args else None
    return request_config