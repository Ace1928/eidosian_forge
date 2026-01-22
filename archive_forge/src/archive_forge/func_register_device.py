from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def register_device(self, identity_pool_id, identity_id, platform, token):
    """
        Registers a device to receive push sync notifications.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. Here, the ID of the pool that the identity belongs to.

        :type identity_id: string
        :param identity_id: The unique ID for this identity.

        :type platform: string
        :param platform: The SNS platform type (e.g. GCM, SDM, APNS,
            APNS_SANDBOX).

        :type token: string
        :param token: The push token.

        """
    uri = '/identitypools/{0}/identity/{1}/device'.format(identity_pool_id, identity_id)
    params = {'Platform': platform, 'Token': token}
    headers = {}
    query_params = {}
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)