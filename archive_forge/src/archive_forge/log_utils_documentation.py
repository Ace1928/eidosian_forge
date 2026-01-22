from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
Returns whether to continue polling the logs.

    Returns False only once we've checked the job and it is finished; we only
    check whether the job is finished once we've gone >1 interval without
    getting any new logs.

    Args:
      periods_without_logs: integer number of empty polls.

    Returns:
      True if we haven't tried polling more than once or if job is not finished.
    