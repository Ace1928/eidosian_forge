from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def set_identity_pool_configuration(self, identity_pool_id, push_sync=None):
    """
        Sets the necessary configuration for push sync.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. This is the ID of the pool to modify.

        :type push_sync: dict
        :param push_sync: Configuration options to be applied to the identity
            pool.

        """
    uri = '/identitypools/{0}/configuration'.format(identity_pool_id)
    params = {}
    headers = {}
    query_params = {}
    if push_sync is not None:
        params['PushSync'] = push_sync
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)