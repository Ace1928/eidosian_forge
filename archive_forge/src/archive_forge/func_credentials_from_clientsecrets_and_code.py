import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@_helpers.positional(3)
def credentials_from_clientsecrets_and_code(filename, scope, code, message=None, redirect_uri='postmessage', http=None, cache=None, device_uri=None):
    """Returns OAuth2Credentials from a clientsecrets file and an auth code.

    Will create the right kind of Flow based on the contents of the
    clientsecrets file or will raise InvalidClientSecretsError for unknown
    types of Flows.

    Args:
        filename: string, File name of clientsecrets.
        scope: string or iterable of strings, scope(s) to request.
        code: string, An authorization code, most likely passed down from
              the client
        message: string, A friendly string to display to the user if the
                 clientsecrets file is missing or invalid. If message is
                 provided then sys.exit will be called in the case of an error.
                 If message in not provided then
                 clientsecrets.InvalidClientSecretsError will be raised.
        redirect_uri: string, this is generally set to 'postmessage' to match
                      the redirect_uri that the client specified
        http: httplib2.Http, optional http instance to use to do the fetch
        cache: An optional cache service client that implements get() and set()
               methods. See clientsecrets.loadfile() for details.
        device_uri: string, OAuth 2.0 device authorization endpoint
        pkce: boolean, default: False, Generate and include a "Proof Key
              for Code Exchange" (PKCE) with your authorization and token
              requests. This adds security for installed applications that
              cannot protect a client_secret. See RFC 7636 for details.
        code_verifier: bytestring or None, default: None, parameter passed
                       as part of the code exchange when pkce=True. If
                       None, a code_verifier will automatically be
                       generated as part of step1_get_authorize_url(). See
                       RFC 7636 for details.

    Returns:
        An OAuth2Credentials object.

    Raises:
        FlowExchangeError: if the authorization code cannot be exchanged for an
                           access token
        UnknownClientSecretsFlowError: if the file describes an unknown kind
                                       of Flow.
        clientsecrets.InvalidClientSecretsError: if the clientsecrets file is
                                                 invalid.
    """
    flow = flow_from_clientsecrets(filename, scope, message=message, cache=cache, redirect_uri=redirect_uri, device_uri=device_uri)
    credentials = flow.step2_exchange(code, http=http)
    return credentials