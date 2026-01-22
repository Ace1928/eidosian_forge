from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.kmsinventory.v1 import kmsinventory_v1_messages as messages
Returns aggregate information about the resources protected by the given Cloud KMS CryptoKey. Only resources within the same Cloud organization as the key will be returned. The project that holds the key must be part of an organization in order for this call to succeed.

      Args:
        request: (KmsinventoryProjectsLocationsKeyRingsCryptoKeysGetProtectedResourcesSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProtectedResourcesSummary) The response message.
      