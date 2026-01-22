import urllib.parse as urlparse
from keystoneauth1 import plugin
from keystoneclient import base
from keystoneclient.v3.contrib.oauth1 import utils
Authorize a request token with specific roles.

        Utilize Identity API operation:
        PUT /OS-OAUTH1/authorize/$request_token_id

        :param request_token: a request token that will be authorized, and
            can be exchanged for an access token.
        :param roles: a list of roles, that will be delegated to the user.
        