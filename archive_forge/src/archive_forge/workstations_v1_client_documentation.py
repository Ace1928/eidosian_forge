from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1 import workstations_v1_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (WorkstationsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      