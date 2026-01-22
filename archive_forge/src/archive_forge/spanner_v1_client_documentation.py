from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
Return available scans given a Database-specific resource name.

      Args:
        request: (SpannerScansListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScansResponse) The response message.
      