import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def list_identities(self, identity_pool_id, max_results, next_token=None):
    """
        Lists the identities in a pool.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type max_results: integer
        :param max_results: The maximum number of identities to return.

        :type next_token: string
        :param next_token: A pagination token.

        """
    params = {'IdentityPoolId': identity_pool_id, 'MaxResults': max_results}
    if next_token is not None:
        params['NextToken'] = next_token
    return self.make_request(action='ListIdentities', body=json.dumps(params))