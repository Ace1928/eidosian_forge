import datetime
import json
from google.auth import external_account_authorized_user
import google.oauth2.credentials
import requests_oauthlib
def session_from_client_secrets_file(client_secrets_file, scopes, **kwargs):
    """Creates a :class:`requests_oauthlib.OAuth2Session` instance from a
    Google-format client secrets file.

    Args:
        client_secrets_file (str): The path to the `client secrets`_ .json
            file.
        scopes (Sequence[str]): The list of scopes to request during the
            flow.
        kwargs: Any additional parameters passed to
            :class:`requests_oauthlib.OAuth2Session`

    Returns:
        Tuple[requests_oauthlib.OAuth2Session, Mapping[str, Any]]: The new
            oauthlib session and the validated client configuration.

    .. _client secrets:
        https://github.com/googleapis/google-api-python-client/blob/main/docs/client-secrets.md
    """
    with open(client_secrets_file, 'r') as json_file:
        client_config = json.load(json_file)
    return session_from_client_config(client_config, scopes, **kwargs)