from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
Returns information, such as supported Azure regions and Kubernetes versions, on a given Google Cloud location.

      Args:
        request: (GkemulticloudProjectsLocationsGetAzureServerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureServerConfig) The response message.
      