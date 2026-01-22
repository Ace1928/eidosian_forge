import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def set_platform_application_attributes(self, platform_application_arn=None, attributes=None):
    """
        The `SetPlatformApplicationAttributes` action sets the
        attributes of the platform application object for the
        supported push notification services, such as APNS and GCM.
        For more information, see `Using Amazon SNS Mobile Push
        Notifications`_.

        :type platform_application_arn: string
        :param platform_application_arn: PlatformApplicationArn for
            SetPlatformApplicationAttributes action.

        :type attributes: map
        :param attributes:
        A map of the platform application attributes. Attributes in this map
            include the following:


        + `PlatformCredential` -- The credential received from the notification
              service. For APNS/APNS_SANDBOX, PlatformCredential is "private
              key". For GCM, PlatformCredential is "API key". For ADM,
              PlatformCredential is "client secret".
        + `PlatformPrincipal` -- The principal received from the notification
              service. For APNS/APNS_SANDBOX, PlatformPrincipal is "SSL
              certificate". For GCM, PlatformPrincipal is not applicable. For
              ADM, PlatformPrincipal is "client id".
        + `EventEndpointCreated` -- Topic ARN to which EndpointCreated event
              notifications should be sent.
        + `EventEndpointDeleted` -- Topic ARN to which EndpointDeleted event
              notifications should be sent.
        + `EventEndpointUpdated` -- Topic ARN to which EndpointUpdate event
              notifications should be sent.
        + `EventDeliveryFailure` -- Topic ARN to which DeliveryFailure event
              notifications should be sent upon Direct Publish delivery failure
              (permanent) to one of the application's endpoints.

        """
    params = {}
    if platform_application_arn is not None:
        params['PlatformApplicationArn'] = platform_application_arn
    if attributes is not None:
        self._build_dict_as_list_params(params, attributes, 'Attributes')
    return self._make_request(action='SetPlatformApplicationAttributes', params=params)