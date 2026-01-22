from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core.console import console_io
Validates that the VPC connector can be safely removed.

  Does nothing if 'clear_vpc_connector' is not present in args with value True.

  Args:
    service: A Cloud Run service object.
    args: Namespace object containing the specified command line arguments.

  Raises:
    exceptions.ConfigurationError: If the command cannot prompt and
      VPC egress is set to 'all' or 'all-traffic'.
    console_io.OperationCancelledError: If the user answers no to the
      confirmation prompt.
  