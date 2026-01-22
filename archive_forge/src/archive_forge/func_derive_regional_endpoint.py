from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from six import text_type
from six.moves.urllib import parse
def derive_regional_endpoint(endpoint, region):
    """Parse the endpoint and add region to it.

  Args:
    endpoint: The url endpoint of the API.
    region: The region for which the endpoint is required.

  Returns:
    regional endpoint for the provided region.
  """
    scheme, netloc, path, params, query, fragment = [text_type(el) for el in parse.urlparse(endpoint)]
    elem = netloc.split('-')
    elem.insert(len(elem) - 1, region)
    netloc = '-'.join(elem)
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))