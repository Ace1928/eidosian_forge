from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
Parse api options from the file and add them to type_provider.

  Args:
    messages: The API message to use.
    options_file: String path expression pointing to a type-provider options
      file.
    type_provider: A TypeProvider message on which the options will be set.

  Returns:
    The type_provider after applying changes.
  Raises:
    exceptions.ConfigError: the api options file couldn't be parsed as yaml
  