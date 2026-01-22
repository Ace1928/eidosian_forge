import datetime
import json
from google.auth import external_account_authorized_user
import google.oauth2.credentials
import requests_oauthlib
def session_from_client_config(client_config, scopes, **kwargs):
    """Creates a :class:`requests_oauthlib.OAuth2Session` from client
    configuration loaded from a Google-format client secrets file.

    Args:
        client_config (Mapping[str, Any]): The client
            configuration in the Google `client secrets`_ format.
        scopes (Sequence[str]): The list of scopes to request during the
            flow.
        kwargs: Any additional parameters passed to
            :class:`requests_oauthlib.OAuth2Session`

    Raises:
        ValueError: If the client configuration is not in the correct
            format.

    Returns:
        Tuple[requests_oauthlib.OAuth2Session, Mapping[str, Any]]: The new
            oauthlib session and the validated client configuration.

    .. _client secrets:
        https://github.com/googleapis/google-api-python-client/blob/main/docs/client-secrets.md
    """
    if 'web' in client_config:
        config = client_config['web']
    elif 'installed' in client_config:
        config = client_config['installed']
    else:
        raise ValueError('Client secrets must be for a web or installed app.')
    if not _REQUIRED_CONFIG_KEYS.issubset(config.keys()):
        raise ValueError('Client secrets is not in the correct format.')
    session = requests_oauthlib.OAuth2Session(client_id=config['client_id'], scope=scopes, **kwargs)
    return (session, client_config)