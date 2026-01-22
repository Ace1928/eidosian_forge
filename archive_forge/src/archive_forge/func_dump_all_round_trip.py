from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
def dump_all_round_trip(documents, stream=None, **kwargs):
    """Dumps multiple YAML documents to the stream using the RoundTripDumper.

  Args:
    documents: An iterable of YAML serializable Python objects to dump.
    stream: The stream to write the data to or None to return it as a string.
    **kwargs: Other arguments to the dump method.

  Returns:
    The string representation of the YAML data if stream is None.
  """
    return yaml.dump_all(documents, stream=stream, default_flow_style=False, indent=2, Dumper=yaml.RoundTripDumper, **kwargs)