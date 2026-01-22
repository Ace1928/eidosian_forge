from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
Service producers use this method to validate if the consumer provided network, project and requested range are valid. This allows them to use a fail-fast mechanism for consumer requests, and not have to wait for AddSubnetwork operation completion to determine if user request is invalid.

      Args:
        request: (ServicenetworkingServicesValidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateConsumerConfigResponse) The response message.
      