from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
def convert_to_block_text(data):
    """This processes the given dict or list so it will render as block text.

  By default, the yaml dumper will write multiline strings out as a double
  quoted string that just includes '\\n'. Calling this on the data strucuture
  will make it use the '|-' notation.

  Args:
    data: {} or [], The data structure to process.
  """
    yaml.scalarstring.walk_tree(data)