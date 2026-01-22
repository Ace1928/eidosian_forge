from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
Initialize an ArgDictFile.

    Args:
      key_type: (str)->str, A function to apply to each of the dict keys.
      value_type: (str)->str, A function to apply to each of the dict values.
    