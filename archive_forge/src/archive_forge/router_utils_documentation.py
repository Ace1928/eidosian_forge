from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
Initializes the instance adapting the error message provided.

    Args:
      messages: API messages holder.
      resource_class: The class of the resource where the ip range is not found.
      error_message: The error message to be formatted with resource_class and
        ip_range.
      ip_range: The ip range that is not found in a resource.
    