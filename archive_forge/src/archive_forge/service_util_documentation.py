from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import text
import six
Format a ServiceNotFoundError.

    Args:
      requested_services: list of str, IDs of services that were not found.
      all_services: list of str, IDs of all available services

    Returns:
      ServicesNotFoundError, error with properly formatted message
    