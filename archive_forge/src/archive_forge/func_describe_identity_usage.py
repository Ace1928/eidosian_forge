from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def describe_identity_usage(self, identity_pool_id, identity_id):
    """
        Gets usage information for an identity, including number of
        datasets and data usage.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        :type identity_id: string
        :param identity_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        """
    uri = '/identitypools/{0}/identities/{1}'.format(identity_pool_id, identity_id)
    return self.make_request('GET', uri, expected_status=200)