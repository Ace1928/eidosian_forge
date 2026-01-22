from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc.shared_messages import autotuning_config_factory as standard_autotuning_config_factory
Builds a RuntimeConfig message.

    Build a RuntimeConfig message instance according to user settings. Returns
    None if all fields are None.

    Args:
      args: Parsed arguments.

    Returns:
      RuntimeConfig: A RuntimeConfig message instance. This function returns
      None if all fields are None.
    