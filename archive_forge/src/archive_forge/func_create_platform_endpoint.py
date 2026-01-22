import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def create_platform_endpoint(self, platform_application_arn=None, token=None, custom_user_data=None, attributes=None):
    """
        The `CreatePlatformEndpoint` creates an endpoint for a device
        and mobile app on one of the supported push notification
        services, such as GCM and APNS. `CreatePlatformEndpoint`
        requires the PlatformApplicationArn that is returned from
        `CreatePlatformApplication`. The EndpointArn that is returned
        when using `CreatePlatformEndpoint` can then be used by the
        `Publish` action to send a message to a mobile app or by the
        `Subscribe` action for subscription to a topic. For more
        information, see `Using Amazon SNS Mobile Push
        Notifications`_.

        :type platform_application_arn: string
        :param platform_application_arn: PlatformApplicationArn returned from
            CreatePlatformApplication is used to create a an endpoint.

        :type token: string
        :param token: Unique identifier created by the notification service for
            an app on a device. The specific name for Token will vary,
            depending on which notification service is being used. For example,
            when using APNS as the notification service, you need the device
            token. Alternatively, when using GCM or ADM, the device token
            equivalent is called the registration ID.

        :type custom_user_data: string
        :param custom_user_data: Arbitrary user data to associate with the
            endpoint. SNS does not use this data. The data must be in UTF-8
            format and less than 2KB.

        :type attributes: map
        :param attributes: For a list of attributes, see
            `SetEndpointAttributes`_.

        """
    params = {}
    if platform_application_arn is not None:
        params['PlatformApplicationArn'] = platform_application_arn
    if token is not None:
        params['Token'] = token
    if custom_user_data is not None:
        params['CustomUserData'] = custom_user_data
    if attributes is not None:
        self._build_dict_as_list_params(params, attributes, 'Attributes')
    return self._make_request(action='CreatePlatformEndpoint', params=params)