from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
def get_generation_number_from_object_name(scheme, object_name):
    """Parses object_name into generation number and object name without trailing # symbol.

  Aplicable for gs:// and s3:// style objects, if the object_name does not
  contain a
  generation number, it will return the object_name as is.

  Args:
    scheme (str): Scheme of URL such as gs and s3.
    object_name (str): Name of a Cloud storage object in the form of object_name
      or object_name#generation_number.

  Returns:
    Object name and generation number if available.
  """
    if scheme == ProviderPrefix.GCS:
        pattern_to_match = GS_GENERATION_REGEX
        group_name = 'generation'
    elif scheme == ProviderPrefix.S3:
        pattern_to_match = S3_VERSION_REGEX
        group_name = 'version_id'
    else:
        return (object_name, None)
    generation_match = pattern_to_match.match(object_name)
    if generation_match is not None:
        return (generation_match.group('object'), generation_match.group(group_name))
    return (object_name, None)