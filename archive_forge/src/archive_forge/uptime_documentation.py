from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
Updates a Stackdriver uptime check.

    If fields is not specified, then the uptime check is replaced entirely. If
    fields are specified, then only the fields are replaced.

    Args:
      uptime_check_ref: resources.Resource, Resource reference to the
        uptime_check to be updated.
      uptime_check: Uptime Check, The uptime_check message to be sent with the
        request.
      fields: str, Comma separated list of field masks.

    Returns:
      Uptime Check, The updated Uptime Check.
    