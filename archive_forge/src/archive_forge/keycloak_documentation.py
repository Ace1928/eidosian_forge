import logging
import os
import pprint
import urllib
import requests
from mistralclient import auth
Performs authentication using Keycloak OpenID Protocol.

        :param req: Request dict containing list of parameters required
            for Keycloak authentication.

            * auth_url: Base authentication url of KeyCloak server (e.g.
                "https://my.keycloak:8443/auth"
            * client_id: Client ID (according to OpenID Connect protocol).
            * client_secret: Client secret (according to OpenID Connect
                protocol).
            * project_name: KeyCloak realm name.
            * username: User name (Optional, if None then access_token must be
                provided).
            * api_key: Password (Optional).
            * access_token: Access token. If passed, username and password are
                not used and this method just validates the token and refreshes
                it if needed (Optional, if None then username must be
                provided).
            * cacert: SSL certificate file (Optional).
            * insecure: If True, SSL certificate is not verified (Optional).

        :param session: Keystone session object. Not used by this plugin.

        