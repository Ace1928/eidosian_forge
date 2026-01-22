from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.ondemandscanning import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import ondemandscanning_util as ods_util
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import platforms
import six
Runs local extraction then calls ODS with the results.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      AnalyzePackages operation.

    Raises:
      UnsupportedOS: when the command is run on a Windows machine.
    