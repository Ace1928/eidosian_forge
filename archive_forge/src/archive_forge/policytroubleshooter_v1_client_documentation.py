from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v1 import policytroubleshooter_v1_messages as messages
Checks whether a principal has a specific permission for a specific resource, and explains why the principal does or does not have that permission.

      Args:
        request: (GoogleCloudPolicytroubleshooterV1TroubleshootIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterV1TroubleshootIamPolicyResponse) The response message.
      