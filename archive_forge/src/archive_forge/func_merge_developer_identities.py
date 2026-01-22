import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def merge_developer_identities(self, source_user_identifier, destination_user_identifier, developer_provider_name, identity_pool_id):
    """
        Merges two users having different `IdentityId`s, existing in
        the same identity pool, and identified by the same developer
        provider. You can use this action to request that discrete
        users be merged and identified as a single user in the Cognito
        environment. Cognito associates the given source user (
        `SourceUserIdentifier`) with the `IdentityId` of the
        `DestinationUserIdentifier`. Only developer-authenticated
        users can be merged. If the users to be merged are associated
        with the same public provider, but as two different users, an
        exception will be thrown.

        :type source_user_identifier: string
        :param source_user_identifier: User identifier for the source user. The
            value should be a `DeveloperUserIdentifier`.

        :type destination_user_identifier: string
        :param destination_user_identifier: User identifier for the destination
            user. The value should be a `DeveloperUserIdentifier`.

        :type developer_provider_name: string
        :param developer_provider_name: The "domain" by which Cognito will
            refer to your users. This is a (pseudo) domain name that you
            provide while creating an identity pool. This name acts as a
            placeholder that allows your backend and the Cognito service to
            communicate about the developer provider. For the
            `DeveloperProviderName`, you can use letters as well as period (.),
            underscore (_), and dash (-).

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        """
    params = {'SourceUserIdentifier': source_user_identifier, 'DestinationUserIdentifier': destination_user_identifier, 'DeveloperProviderName': developer_provider_name, 'IdentityPoolId': identity_pool_id}
    return self.make_request(action='MergeDeveloperIdentities', body=json.dumps(params))