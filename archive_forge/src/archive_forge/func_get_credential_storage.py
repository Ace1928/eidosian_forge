import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
@util.positional(4)
def get_credential_storage(filename, client_id, user_agent, scope, warn_on_readonly=True):
    """Get a Storage instance for a credential.

    Args:
        filename: The JSON file storing a set of credentials
        client_id: The client_id for the credential
        user_agent: The user agent for the credential
        scope: string or iterable of strings, Scope(s) being requested
        warn_on_readonly: if True, log a warning if the store is readonly

    Returns:
        An object derived from client.Storage for getting/setting the
        credential.
    """
    key = {'clientId': client_id, 'userAgent': user_agent, 'scope': util.scopes_to_string(scope)}
    return get_credential_storage_custom_key(filename, key, warn_on_readonly=warn_on_readonly)