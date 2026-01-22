from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.dataproc.poller import (
from googlecloudsdk.core import log
Handles errors.

    Error handling for sessions. This happen after the session reaches one of
    the complete states.

    Overrides.

    Args:
      session: The session resource.

    Returns:
      None. The result is directly output to log.err.

    Raises:
      OperationTimeoutError: When waiter timed out.
      OperationError: When remote session creation is failed.
    