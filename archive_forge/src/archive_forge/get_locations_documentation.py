from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import flags
This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. From `Args`, we extract command line
        arguments

    Returns:
      List of dict values for locations of instance
    