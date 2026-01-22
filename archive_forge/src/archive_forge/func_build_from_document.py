from __future__ import absolute_import
import six
from six.moves import zip
from six import BytesIO
from six.moves import http_client
from six.moves.urllib.parse import urlencode, urlparse, urljoin, urlunparse, parse_qsl
import copy
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import json
import keyword
import logging
import mimetypes
import os
import re
import httplib2
import uritemplate
import google.api_core.client_options
from google.auth.transport import mtls
from google.auth.exceptions import MutualTLSChannelError
from googleapiclient import _auth
from googleapiclient import mimeparse
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidJsonError
from googleapiclient.errors import MediaUploadSizeError
from googleapiclient.errors import UnacceptableMimeTypeError
from googleapiclient.errors import UnknownApiNameOrVersion
from googleapiclient.errors import UnknownFileType
from googleapiclient.http import build_http
from googleapiclient.http import BatchHttpRequest
from googleapiclient.http import HttpMock
from googleapiclient.http import HttpMockSequence
from googleapiclient.http import HttpRequest
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaUpload
from googleapiclient.model import JsonModel
from googleapiclient.model import MediaModel
from googleapiclient.model import RawModel
from googleapiclient.schema import Schemas
from googleapiclient._helpers import _add_query_parameter
from googleapiclient._helpers import positional
@positional(1)
def build_from_document(service, base=None, future=None, http=None, developerKey=None, model=None, requestBuilder=HttpRequest, credentials=None, client_options=None, adc_cert_path=None, adc_key_path=None):
    """Create a Resource for interacting with an API.

  Same as `build()`, but constructs the Resource object from a discovery
  document that is it given, as opposed to retrieving one over HTTP.

  Args:
    service: string or object, the JSON discovery document describing the API.
      The value passed in may either be the JSON string or the deserialized
      JSON.
    base: string, base URI for all HTTP requests, usually the discovery URI.
      This parameter is no longer used as rootUrl and servicePath are included
      within the discovery document. (deprecated)
    future: string, discovery document with future capabilities (deprecated).
    http: httplib2.Http, An instance of httplib2.Http or something that acts
      like it that HTTP requests will be made through.
    developerKey: string, Key for controlling API usage, generated
      from the API Console.
    model: Model class instance that serializes and de-serializes requests and
      responses.
    requestBuilder: Takes an http request and packages it up to be executed.
    credentials: oauth2client.Credentials or
      google.auth.credentials.Credentials, credentials to be used for
      authentication.
    client_options: Mapping object or google.api_core.client_options, client
      options to set user options on the client.
      (1) The API endpoint should be set through client_options. If API endpoint
      is not set, `GOOGLE_API_USE_MTLS_ENDPOINT` environment variable can be used
      to control which endpoint to use.
      (2) client_cert_source is not supported, client cert should be provided using
      client_encrypted_cert_source instead. In order to use the provided client
      cert, `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be
      set to `true`.
      More details on the environment variables are here:
      https://google.aip.dev/auth/4114
    adc_cert_path: str, client certificate file path to save the application
      default client certificate for mTLS. This field is required if you want to
      use the default client certificate. `GOOGLE_API_USE_CLIENT_CERTIFICATE`
      environment variable must be set to `true` in order to use this field,
      otherwise this field is ignored.
      More details on the environment variables are here:
      https://google.aip.dev/auth/4114
    adc_key_path: str, client encrypted private key file path to save the
      application default client encrypted private key for mTLS. This field is
      required if you want to use the default client certificate.
      `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be set to
      `true` in order to use this field, otherwise this field is ignored.
      More details on the environment variables are here:
      https://google.aip.dev/auth/4114

  Returns:
    A Resource object with methods for interacting with the service.

  Raises:
    google.auth.exceptions.MutualTLSChannelError: if there are any problems
      setting up mutual TLS channel.
  """
    client_options = get_client_options(client_options)
    service = _get_service(service)
    if http is not None:
        banned_options = [(credentials, 'credentials'), (client_options.credentials_file, 'client_options.credentials_file')]
        for option, name in banned_options:
            if option is not None:
                raise ValueError('Arguments http and {} are mutually exclusive'.format(name))
    if 'rootUrl' not in service and isinstance(http, (HttpMock, HttpMockSequence)):
        logger.error('You are using HttpMock or HttpMockSequence without' + 'having the service discovery doc in cache. Try calling ' + 'build() without mocking once first to populate the ' + 'cache.')
        raise InvalidJsonError()
    base = urljoin(service['rootUrl'], service['servicePath'])
    if client_options.api_endpoint:
        base = client_options.api_endpoint
    schema = Schemas(service)
    if http is None:
        scopes = list(service.get('auth', {}).get('oauth2', {}).get('scopes', {}).keys())
        if scopes and (not developerKey):
            if client_options.credentials_file and credentials:
                raise google.api_core.exceptions.DuplicateCredentialArgs('client_options.credentials_file and credentials are mutually exclusive.')
            if client_options.credentials_file:
                credentials = _auth.credentials_from_file(client_options.credentials_file, scopes=client_options.scopes, quota_project_id=client_options.quota_project_id)
            if credentials is None:
                credentials = _auth.default_credentials(scopes=client_options.scopes, quota_project_id=client_options.quota_project_id)
            if not client_options.scopes:
                credentials = _auth.with_scopes(credentials, scopes)
        if credentials:
            http = _auth.authorized_http(credentials)
        else:
            http = build_http()
        use_client_cert = os.getenv(GOOGLE_API_USE_CLIENT_CERTIFICATE, 'false')
        if use_client_cert not in ('true', 'false'):
            raise MutualTLSChannelError('Unsupported GOOGLE_API_USE_CLIENT_CERTIFICATE value. Accepted values: true, false')
        client_cert_used = False
        if use_client_cert == 'true':
            client_cert_used = add_mtls_creds(http, client_options, adc_cert_path, adc_key_path)
        if not client_options or not client_options.api_endpoint:
            mtls_endpoint = _get_mtls_endpoint(service)
            if mtls_endpoint:
                use_mtls_endpoint = os.getenv(GOOGLE_API_USE_MTLS_ENDPOINT, 'auto')
                if use_mtls_endpoint not in ('never', 'auto', 'always'):
                    raise MutualTLSChannelError('Unsupported GOOGLE_API_USE_MTLS_ENDPOINT value. Accepted values: never, auto, always')
                if use_mtls_endpoint == 'always' or (use_mtls_endpoint == 'auto' and client_cert_used):
                    base = mtls_endpoint
    if model is None:
        features = service.get('features', [])
        model = JsonModel('dataWrapper' in features)
    return Resource(http=http, baseUrl=base, model=model, developerKey=developerKey, requestBuilder=requestBuilder, resourceDesc=service, rootDesc=service, schema=schema)