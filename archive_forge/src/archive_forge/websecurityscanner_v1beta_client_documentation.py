from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.websecurityscanner.v1beta import websecurityscanner_v1beta_messages as messages
Start a ScanRun according to the given ScanConfig.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScanRun) The response message.
      