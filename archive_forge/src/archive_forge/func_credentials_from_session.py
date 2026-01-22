import datetime
import json
from google.auth import external_account_authorized_user
import google.oauth2.credentials
import requests_oauthlib
def credentials_from_session(session, client_config=None):
    """Creates :class:`google.oauth2.credentials.Credentials` from a
    :class:`requests_oauthlib.OAuth2Session`.

    :meth:`fetch_token` must be called on the session before before calling
    this. This uses the session's auth token and the provided client
    configuration to create :class:`google.oauth2.credentials.Credentials`.
    This allows you to use the credentials from the session with Google
    API client libraries.

    Args:
        session (requests_oauthlib.OAuth2Session): The OAuth 2.0 session.
        client_config (Mapping[str, Any]): The subset of the client
            configuration to use. For example, if you have a web client
            you would pass in `client_config['web']`.

    Returns:
        google.oauth2.credentials.Credentials: The constructed credentials.

    Raises:
        ValueError: If there is no access token in the session.
    """
    client_config = client_config if client_config is not None else {}
    if not session.token:
        raise ValueError('There is no access token for this session, did you call fetch_token?')
    if '3pi' in client_config:
        credentials = external_account_authorized_user.Credentials(token=session.token['access_token'], refresh_token=session.token.get('refresh_token'), token_url=client_config.get('token_uri'), client_id=client_config.get('client_id'), client_secret=client_config.get('client_secret'), token_info_url=client_config.get('token_info_url'), scopes=session.scope)
    else:
        credentials = google.oauth2.credentials.Credentials(session.token['access_token'], refresh_token=session.token.get('refresh_token'), id_token=session.token.get('id_token'), token_uri=client_config.get('token_uri'), client_id=client_config.get('client_id'), client_secret=client_config.get('client_secret'), scopes=session.scope, granted_scopes=session.token.get('scope'))
    credentials.expiry = datetime.datetime.utcfromtimestamp(session.token['expires_at'])
    return credentials