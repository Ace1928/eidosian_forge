from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def describe_identity_pool_usage(self, identity_pool_id):
    """
        Gets usage details (for example, data storage) about a
        particular identity pool.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        """
    uri = '/identitypools/{0}'.format(identity_pool_id)
    return self.make_request('GET', uri, expected_status=200)