from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def check_if_use_gsutil_style(args):
    """Check if format output using gsutil style.

  Args:
    args (object): User input arguments.

  Returns:
    use_gsutil_style (bool): True if format with gsutil style.
  """
    if args.format:
        if args.format != 'gsutil':
            raise errors.Error('The only valid format value for ls and du is "gsutil" (e.g. "--format=gsutil"). See other flags and commands for additional formatting options.')
        use_gsutil_style = True
        args.format = None
    else:
        use_gsutil_style = properties.VALUES.storage.run_by_gsutil_shim.GetBool()
    return use_gsutil_style