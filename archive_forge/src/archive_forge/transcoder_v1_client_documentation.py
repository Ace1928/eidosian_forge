from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.transcoder.v1 import transcoder_v1_messages as messages
Lists jobs in the specified region.

      Args:
        request: (TranscoderProjectsLocationsJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobsResponse) The response message.
      