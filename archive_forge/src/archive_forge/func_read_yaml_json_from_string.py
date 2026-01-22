from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import yaml
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import files
import six
def read_yaml_json_from_string(string, source_path=None):
    """Converts JSON or YAML stream to an in-memory dict."""
    current_error_string = 'Found invalid JSON/YAML'
    if source_path:
        current_error_string += ' in {}'.format(source_path)
    try:
        parsed = yaml.load(string)
        if isinstance(parsed, dict) or isinstance(parsed, list):
            return parsed
    except yaml.YAMLParseError as e:
        current_error_string += '\n\nYAML Error: {}'.format(six.text_type(e))
    try:
        return json.loads(string)
    except json.JSONDecodeError as e:
        current_error_string += '\n\nJSON Error: {}'.format(six.text_type(e))
    raise errors.InvalidUrlError(current_error_string)