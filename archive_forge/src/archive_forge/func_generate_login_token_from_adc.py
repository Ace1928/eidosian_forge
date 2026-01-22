from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import credentials
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import credentials as google_auth_creds
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
from googlecloudsdk.core.credentials import store as c_store
import six
def generate_login_token_from_adc(scopes):
    """Genearete a down-coped access token with given scopes for IAM DB authentication from application default credentials.

  Args:
    scopes: scopes to be included in the down-scoped token.

  Returns:
    Down-scoped access token.
  """
    try:
        creds, _ = c_creds.GetGoogleAuthDefault().default(scopes=scopes)
    except google_auth_exceptions.DefaultCredentialsError as e:
        log.debug(e, exc_info=True)
        raise c_exc.ToolException(six.text_type(e))
    creds = _downscope_credential(creds, scopes)
    if isinstance(creds, google_auth_creds.Credentials):
        creds = c_google_auth.Credentials.FromGoogleAuthUserCredentials(creds)
    with c_store.HandleGoogleAuthCredentialsRefreshError(for_adc=True):
        creds.refresh(requests.GoogleAuthRequest())
    return creds