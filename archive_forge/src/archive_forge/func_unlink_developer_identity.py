import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def unlink_developer_identity(self, identity_id, identity_pool_id, developer_provider_name, developer_user_identifier):
    """
        Unlinks a `DeveloperUserIdentifier` from an existing identity.
        Unlinked developer users will be considered new identities
        next time they are seen. If, for a given Cognito identity, you
        remove all federated identities as well as the developer user
        identifier, the Cognito identity becomes inaccessible.

        :type identity_id: string
        :param identity_id: A unique identifier in the format REGION:GUID.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type developer_provider_name: string
        :param developer_provider_name: The "domain" by which Cognito will
            refer to your users.

        :type developer_user_identifier: string
        :param developer_user_identifier: A unique ID used by your backend
            authentication process to identify a user.

        """
    params = {'IdentityId': identity_id, 'IdentityPoolId': identity_pool_id, 'DeveloperProviderName': developer_provider_name, 'DeveloperUserIdentifier': developer_user_identifier}
    return self.make_request(action='UnlinkDeveloperIdentity', body=json.dumps(params))