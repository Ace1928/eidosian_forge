from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six.moves.http_client
Performs a storage size decrease of a Cloud SQL instance.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      A dict object representing the operations resource describing the perform
      storage shrink operation if the decrease was successful.

    Raises:
      HttpException: A http error response was received while executing api
          request.
      ResourceNotFoundError: The SQL instance wasn't found.
      RequiredArgumentException: A required argument was not supplied by the
      user, such as omitting --root-password on a SQL Server instance.
    