from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
Returns the CliTreeInfo list of all available CLI trees.

  Args:
    directory: The config directory containing the CLI tree modules.

  Raises:
    CliTreeVersionError: loaded tree version mismatch
    ImportModuleError: import errors

  Returns:
    The CLI tree.
  