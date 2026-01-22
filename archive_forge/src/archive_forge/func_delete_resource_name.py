from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
@staticmethod
def delete_resource_name(metric):
    """Delete the name fields from metric message.

    Args:
      metric: The quota metric message.

    Returns:
      The updated metric message.
    """
    metric.reset('name')
    for l in metric.consumerQuotaLimits:
        l.reset('name')
    return metric