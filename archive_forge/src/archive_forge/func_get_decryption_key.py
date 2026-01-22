from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import collections
import enum
import hashlib
import re
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.cache import function_result_cache
def get_decryption_key(sha256_hash, url_for_missing_key_error=None):
    """Returns a key that matches sha256_hash, or None if no key is found."""
    if _key_store.initialized:
        decryption_key = _key_store.decryption_key_index.get(sha256_hash)
    else:
        decryption_key = None
    if not decryption_key and url_for_missing_key_error:
        raise errors.Error('Missing decryption key with SHA256 hash {}. No decryption key matches object {}.'.format(sha256_hash, url_for_missing_key_error))
    return decryption_key