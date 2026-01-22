from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def probe_access_to_resource(client_id, host, key, path, region, requested_headers, requested_http_verb, requested_parameters, requested_resource):
    """Checks if provided credentials offer appropriate access to a resource.

  Args:
    client_id (str): Email of the service account that makes the request.
    host (str): The endpoint URL for the request.
    key (crypto.PKey): Key for the service account specified by client_id.
    path (str): Of the form `/bucket-name/object-name`. Specifies the resource
      that is targeted by the request.
    region (str): The region of the target resource instance.
    requested_headers (dict[str, str]): Headers used in the user's request.
      These do not need to be passed into the HEAD request performed by this
      function, but they do need to be checked for this function to raise
      appropriate errors for different use cases (e.g. for resumable uploads).
    requested_http_verb (str): Method the user requested.
    requested_parameters (dict[str, str]): URL parameters the user requested.
    requested_resource (resource_reference.Resource): Resource the user
      requested to access.

  Raises:
    errors.Error if the requested resource is not available for the requested
      operation.
  """
    parameters = {}
    if 'userProject' in requested_parameters:
        parameters['userProject'] = requested_parameters['userProject']
    url = get_signed_url(client_id=client_id, duration=60, headers={}, host=host, key=key, verb='HEAD', parameters=parameters, path=path, region=region, delegates=None)
    session = core_requests.GetSession()
    response = session.head(url)
    if response.status_code == 404:
        if requested_http_verb == 'PUT':
            return
        is_resumable_upload = 'x-goog-resumable' in requested_headers
        if is_resumable_upload:
            return
        if requested_resource.storage_url.is_bucket():
            raise errors.Error('Bucket {} does not exist. Please create a bucket with that name before creating a signed URL to access it.'.format(requested_resource.storage_url))
        else:
            raise errors.Error('Object {} does not exist. Please create an object with that name before creating a signed URL to access it.'.format(requested_resource.storage_url))
    elif response.status_code == 403:
        log.warning('{} does not have permissions on {}. Using this link will likely result in a 403 error until at least READ permissions are granted.'.format(client_id, requested_resource.storage_url))
    else:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            raise errors.Error('Expected an HTTP response code of 200 while querying object readability, but received an error: {}'.format(error))