from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from six import text_type
from six.moves.urllib import parse
def base_64_encoding(file_path=None, blueprint_plan_=None):
    """Encodes content of a blueprint plan to base64 bytes.

  Args:
    file_path: The path of the blueprint plan file to be encoded.
    blueprint_plan_: The string of the blueprint json file.

  Returns:
    bytes of the message.
  """
    if blueprint_plan_ is None:
        blueprint_plan_ = files.ReadFileContents(file_path)
    return blueprint_plan_.encode()