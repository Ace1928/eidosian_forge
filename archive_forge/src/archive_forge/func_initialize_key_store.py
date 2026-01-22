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
def initialize_key_store(args):
    """Loads appropriate encryption and decryption keys into memory.

  Prefers values from flags over those from the user's key file. If _key_store
    is not already initialized, creates a _KeyStore instance and stores it in a
    global variable.

  Args:
    args: An object containing flag values from the command surface.
  """
    if _key_store.initialized:
        return
    raw_encryption_key = _get_raw_key(args, 'encryption_key')
    if getattr(args, 'clear_encryption_key', None):
        _key_store.encryption_key = user_request_args_factory.CLEAR
    elif raw_encryption_key:
        _key_store.encryption_key = parse_key(raw_encryption_key)
    raw_keys = [raw_encryption_key]
    raw_decryption_keys = _get_raw_key(args, 'decryption_keys')
    if raw_decryption_keys:
        raw_keys += raw_decryption_keys
    _key_store.decryption_key_index = _index_decryption_keys(raw_keys)
    _key_store.initialized = True