from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.oauth2 import utils as oauth2_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from six.moves import http_client
from six.moves import urllib
Returns the meta-information associated with an OAuth token.

    Args:
      request (google.auth.transport.Request): A callable that makes HTTP
        requests.
      token (str): The OAuth token whose meta-information are to be returned.
      token_type_hint (Optional[str]): The optional token type. The default is
        access_token.

    Returns:
      Mapping: The active token meta-information returned by the introspection
        endpoint.

    Raises:
      InactiveCredentialsError: If the credentials are invalid or expired.
      TokenIntrospectionError: If an error is encountered while calling the
        token introspection endpoint.
    