from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import sys
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import iso_duration
def process_lifecycle(file_path):
    """Converts lifecycle file to Apitools objects."""
    if file_path == user_request_args_factory.CLEAR:
        return None
    lifecycle_dict = metadata_util.cached_read_yaml_json_file(file_path)
    if not lifecycle_dict:
        return None
    messages = apis.GetMessagesModule('storage', 'v1')
    if 'lifecycle' in lifecycle_dict:
        lifecycle_rules_dict = lifecycle_dict['lifecycle']
    else:
        lifecycle_rules_dict = lifecycle_dict
    try:
        return messages_util.DictToMessageWithErrorCheck(lifecycle_rules_dict, messages.Bucket.LifecycleValue)
    except messages_util.DecodeError:
        raise errors.InvalidUrlError('Found invalid JSON/YAML for the lifecycle rule')