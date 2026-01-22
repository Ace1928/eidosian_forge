import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def lookup_developer_identity(self, identity_pool_id, identity_id=None, developer_user_identifier=None, max_results=None, next_token=None):
    """
        Retrieves the `IdentityID` associated with a
        `DeveloperUserIdentifier` or the list of
        `DeveloperUserIdentifier`s associated with an `IdentityId` for
        an existing identity. Either `IdentityID` or
        `DeveloperUserIdentifier` must not be null. If you supply only
        one of these values, the other value will be searched in the
        database and returned as a part of the response. If you supply
        both, `DeveloperUserIdentifier` will be matched against
        `IdentityID`. If the values are verified against the database,
        the response returns both values and is the same as the
        request. Otherwise a `ResourceConflictException` is thrown.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type identity_id: string
        :param identity_id: A unique identifier in the format REGION:GUID.

        :type developer_user_identifier: string
        :param developer_user_identifier: A unique ID used by your backend
            authentication process to identify a user. Typically, a developer
            identity provider would issue many developer user identifiers, in
            keeping with the number of users.

        :type max_results: integer
        :param max_results: The maximum number of identities to return.

        :type next_token: string
        :param next_token: A pagination token. The first call you make will
            have `NextToken` set to null. After that the service will return
            `NextToken` values as needed. For example, let's say you make a
            request with `MaxResults` set to 10, and there are 20 matches in
            the database. The service will return a pagination token as a part
            of the response. This token can be used to call the API again and
            get results starting from the 11th match.

        """
    params = {'IdentityPoolId': identity_pool_id}
    if identity_id is not None:
        params['IdentityId'] = identity_id
    if developer_user_identifier is not None:
        params['DeveloperUserIdentifier'] = developer_user_identifier
    if max_results is not None:
        params['MaxResults'] = max_results
    if next_token is not None:
        params['NextToken'] = next_token
    return self.make_request(action='LookupDeveloperIdentity', body=json.dumps(params))