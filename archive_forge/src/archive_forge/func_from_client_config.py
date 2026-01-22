from base64 import urlsafe_b64encode
import hashlib
import json
import logging
from string import ascii_letters, digits
import webbrowser
import wsgiref.simple_server
import wsgiref.util
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.helpers
@classmethod
def from_client_config(cls, client_config, scopes, **kwargs):
    """Creates a :class:`requests_oauthlib.OAuth2Session` from client
        configuration loaded from a Google-format client secrets file.

        Args:
            client_config (Mapping[str, Any]): The client
                configuration in the Google `client secrets`_ format.
            scopes (Sequence[str]): The list of scopes to request during the
                flow.
            kwargs: Any additional parameters passed to
                :class:`requests_oauthlib.OAuth2Session`

        Returns:
            Flow: The constructed Flow instance.

        Raises:
            ValueError: If the client configuration is not in the correct
                format.

        .. _client secrets:
            https://github.com/googleapis/google-api-python-client/blob/main/docs/client-secrets.md
        """
    if 'web' in client_config:
        client_type = 'web'
    elif 'installed' in client_config:
        client_type = 'installed'
    else:
        raise ValueError('Client secrets must be for a web or installed app.')
    code_verifier = kwargs.pop('code_verifier', None)
    autogenerate_code_verifier = kwargs.pop('autogenerate_code_verifier', None)
    session, client_config = google_auth_oauthlib.helpers.session_from_client_config(client_config, scopes, **kwargs)
    redirect_uri = kwargs.get('redirect_uri', None)
    return cls(session, client_type, client_config, redirect_uri, code_verifier, autogenerate_code_verifier)