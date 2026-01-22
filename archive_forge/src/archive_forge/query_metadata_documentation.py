from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.metastore import operations_util
from googlecloudsdk.api_lib.metastore import services_util as services_api_util
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.metastore import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
Displays the server response to a query.

    This is called higher up the stack to over-write default display behavior.
    What gets displayed depends on the mode in which the query was run.

    Args:
      args: The arguments originally passed to the command.
      result: The output of the command before display.
    