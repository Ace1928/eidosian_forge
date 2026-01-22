import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def get_platform_application_attributes(self, platform_application_arn=None):
    """
        The `GetPlatformApplicationAttributes` action retrieves the
        attributes of the platform application object for the
        supported push notification services, such as APNS and GCM.
        For more information, see `Using Amazon SNS Mobile Push
        Notifications`_.

        :type platform_application_arn: string
        :param platform_application_arn: PlatformApplicationArn for
            GetPlatformApplicationAttributesInput.

        """
    params = {}
    if platform_application_arn is not None:
        params['PlatformApplicationArn'] = platform_application_arn
    return self._make_request(action='GetPlatformApplicationAttributes', params=params)