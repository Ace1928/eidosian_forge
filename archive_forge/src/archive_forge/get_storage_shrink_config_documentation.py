from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
import six.moves.http_client
Displays the minimum storage size to which a Cloud SQL instance can be decreased.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A kind string representing the request run and the minimum storage
      size to which a Cloud SQL instance can be decreased.

    Raises:
      HttpException: A http error response was received while executing api
          request.
      ResourceNotFoundError: The SQL instance isn't found.
    