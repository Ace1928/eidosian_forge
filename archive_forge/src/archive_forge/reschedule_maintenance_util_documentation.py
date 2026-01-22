from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from dateutil import tz
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
Reschedule maintenance for a Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.

  Returns:
    None

  Raises:
    HttpException: An HTTP error response was received while executing API
        request.
    ArgumentError: The schedule_time argument was missing, in an invalid format,
        or not within the reschedule maintenance bounds.
    InvalidStateException: The Cloud SQL instance was not in an appropriate
        state for the requested command.
    ToolException: Any other error that occurred while executing the command.
  