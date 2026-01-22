import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def get_open_id_token_for_developer_identity(self, identity_pool_id, logins, identity_id=None, token_duration=None):
    """
        Registers (or retrieves) a Cognito `IdentityId` and an OpenID
        Connect token for a user authenticated by your backend
        authentication process. Supplying multiple logins will create
        an implicit linked account. You can only specify one developer
        provider as part of the `Logins` map, which is linked to the
        identity pool. The developer provider is the "domain" by which
        Cognito will refer to your users.

        You can use `GetOpenIdTokenForDeveloperIdentity` to create a
        new identity and to link new logins (that is, user credentials
        issued by a public provider or developer provider) to an
        existing identity. When you want to create a new identity, the
        `IdentityId` should be null. When you want to associate a new
        login with an existing authenticated/unauthenticated identity,
        you can do so by providing the existing `IdentityId`. This API
        will create the identity in the specified `IdentityPoolId`.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type identity_id: string
        :param identity_id: A unique identifier in the format REGION:GUID.

        :type logins: map
        :param logins: A set of optional name-value pairs that map provider
            names to provider tokens. Each name-value pair represents a user
            from a public provider or developer provider. If the user is from a
            developer provider, the name-value pair will follow the syntax
            `"developer_provider_name": "developer_user_identifier"`. The
            developer provider is the "domain" by which Cognito will refer to
            your users; you provided this domain while creating/updating the
            identity pool. The developer user identifier is an identifier from
            your backend that uniquely identifies a user. When you create an
            identity pool, you can specify the supported logins.

        :type token_duration: long
        :param token_duration: The expiration time of the token, in seconds.
            You can specify a custom expiration time for the token so that you
            can cache it. If you don't provide an expiration time, the token is
            valid for 15 minutes. You can exchange the token with Amazon STS
            for temporary AWS credentials, which are valid for a maximum of one
            hour. The maximum token duration you can set is 24 hours. You
            should take care in setting the expiration time for a token, as
            there are significant security implications: an attacker could use
            a leaked token to access your AWS resources for the token's
            duration.

        """
    params = {'IdentityPoolId': identity_pool_id, 'Logins': logins}
    if identity_id is not None:
        params['IdentityId'] = identity_id
    if token_duration is not None:
        params['TokenDuration'] = token_duration
    return self.make_request(action='GetOpenIdTokenForDeveloperIdentity', body=json.dumps(params))