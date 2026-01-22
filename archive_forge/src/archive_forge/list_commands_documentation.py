from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import walker_util
Visit each command and group in the CLI command tree.

    Args:
      command: dict, The tree (nested dict) of command/group names.
      prefix: [str], The subcommand arg prefix.
    