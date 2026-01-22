from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def validate_issuer_uri(issuer_uri, auth_method_name):
    """Validates Issuer URI field of OIDC config.

  Args:
    issuer_uri: issuer uri to be validated
    auth_method_name: auth method name that has this field
  """
    url = urllib3.util.parse_url(issuer_uri)
    if url.scheme != 'https':
        raise exceptions.Error('issuerURI is invalid for method [{}]. Scheme is not https.'.format(auth_method_name))
    if url.path is not None and '.well-known/openid-configuration' in url.path:
        raise exceptions.Error('issuerURI is invalid for method [{}]. issuerURI should not contain [{}].'.format(auth_method_name, '.well-known/openid-configuration'))