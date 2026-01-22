from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securedlandingzone.v1beta import securedlandingzone_v1beta_messages as messages
Suspend an overwatch resource. This sets the state to SUSPENDED, and will stop all future response actions.

      Args:
        request: (SecuredlandingzoneOrganizationsLocationsOverwatchesSuspendRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuredlandingzoneV1betaOverwatch) The response message.
      