from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1beta import workstations_v1beta_messages as messages
Updates an existing workstation cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      