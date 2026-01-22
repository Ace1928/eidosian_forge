import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def update_identity_pool(self, identity_pool_id, identity_pool_name, allow_unauthenticated_identities, supported_login_providers=None, developer_provider_name=None, open_id_connect_provider_ar_ns=None):
    """
        Updates a user pool.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type identity_pool_name: string
        :param identity_pool_name: A string that you provide.

        :type allow_unauthenticated_identities: boolean
        :param allow_unauthenticated_identities: TRUE if the identity pool
            supports unauthenticated logins.

        :type supported_login_providers: map
        :param supported_login_providers: Optional key:value pairs mapping
            provider names to provider app IDs.

        :type developer_provider_name: string
        :param developer_provider_name: The "domain" by which Cognito will
            refer to your users.

        :type open_id_connect_provider_ar_ns: list
        :param open_id_connect_provider_ar_ns:

        """
    params = {'IdentityPoolId': identity_pool_id, 'IdentityPoolName': identity_pool_name, 'AllowUnauthenticatedIdentities': allow_unauthenticated_identities}
    if supported_login_providers is not None:
        params['SupportedLoginProviders'] = supported_login_providers
    if developer_provider_name is not None:
        params['DeveloperProviderName'] = developer_provider_name
    if open_id_connect_provider_ar_ns is not None:
        params['OpenIdConnectProviderARNs'] = open_id_connect_provider_ar_ns
    return self.make_request(action='UpdateIdentityPool', body=json.dumps(params))